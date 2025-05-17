import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, \
    recall_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import time


class Trainer:
    """
    Trainer class for training the model
    """

    def __init__(self, model, optimizer, criterion, train_loader, val_loader, device, args, checkpoint_saver=None):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_threshold = 0.5
        self.checkpoint_saver = checkpoint_saver
        self.task_name = args.task
        self.args = args

    def run(self, epochs=100, verbose=True, patience=10):
        """
        if self.wandb_logger is not None:
            self.wandb_logger.watch_model(self.model)
        """
        e = 0
        train_loss = []
        eval_loss = []
        patience_count = 0
        prev_val_loss = 1e8
        while True:
            time_start = time.time()
            tloss = self.train()
            print(f"Epoch {e} costs {time.time() - time_start} s.")
            eloss, metrics = self.eval()
            # verbose
            if verbose and e % 2 == 0:
                print({
                    "epoch": e,
                    "train_loss": tloss,
                    "eval_loss": eloss,
                    "metrics": metrics,
                })

            train_loss.append(tloss)
            eval_loss.append(eloss)

            # save model
            best_model = self.checkpoint_saver.save(
                e, self.model, self.optimizer, eloss
            )

            if eloss < prev_val_loss:
                patience_count = 0
            else:
                patience_count += 1
            prev_val_loss = eloss
            e += 1
            if e > epochs or patience_count > patience or eloss <= 1e-4:
                break
        return (train_loss, eval_loss)

    def train(self):
        self.model.train()
        train_loss = []
        for batch in tqdm(self.train_loader):
            # get batch
            x1 = batch[0].to(self.device)
            x2 = batch[1].to(self.device)
            len1 = batch[2].to(self.device)
            len2 = batch[3].to(self.device)
            y = batch[4].to(self.device)
            # zero the gradients
            self.optimizer.zero_grad()
            logits = self.model(x1, len1, self.args.dataset_type1, x2, len2, self.args.dataset_type2)

            if self.args.task == "multi-relation" or self.args.task == "direction-prediction":
                y = y.type(torch.LongTensor).to(self.device)
            loss = self.criterion(logits, y)
            # save the loss
            train_loss.append(loss.item())
            # backward pass
            loss.backward()

            self.optimizer.step()
        # return the average loss
        return np.mean(train_loss)

    def eval(self):
        self.model.eval()
        val_loss = []
        y_true, y_prob, y_pred, y_logit = [], [], [], []
        y_true_multi = []
        with torch.no_grad():
            for batch in self.val_loader:
                # get batch
                x1 = batch[0].to(self.device)
                x2 = batch[1].to(self.device)
                len1 = batch[2].to(self.device)
                len2 = batch[3].to(self.device)
                y = batch[4].to(self.device)
                y_true.extend(y.cpu().numpy())
                # forward pass
                logits = self.model(x1, len1, self.args.dataset_type1, x2, len2, self.args.dataset_type2)

                if self.args.task == "multi-relation" or self.args.task == "direction-prediction":
                    y = y.type(torch.LongTensor).to(self.device)
                    y_true_multi.extend(label_binarize(y.cpu().numpy(), classes=[i for i in range(self.args.num_classes)]))
                loss = self.criterion(logits, y)
                val_loss.append(loss.item())

                # compute metrics
                if self.args.task == "multi-relation" or self.args.task == "direction-prediction":
                    y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().astype(int))
                    y_prob.extend(logits.cpu().numpy())
                else:
                    out_probs = logits.cpu().numpy()
                    y_logit.extend(logits.cpu().numpy())
                    y_prob.extend(out_probs)
                    y_pred.extend((out_probs > self.best_threshold).astype(int))
        # compute metrics
        if self.task_name == 'relation-prediction':
            # self.best_threshold = thresh_max_f1(y_true=y_true, y_prob=y_prob)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='binary')
            precision = precision_score(y_true, y_pred, average='binary')
            recall = recall_score(y_true, y_pred, average='binary')
            aucroc = roc_auc_score(y_true=y_true, y_score=y_prob)
            auprc = average_precision_score(y_true=y_true, y_score=y_prob)
            cm = confusion_matrix(y_true, y_pred)
            metrics = {'accuracy': accuracy, 'f1': f1, 'aucroc': aucroc,
                       'best_threshold': self.best_threshold, 'auprc': auprc,
                       'precision': precision, 'recall': recall, 'confusion_matrix': cm}
        elif self.task_name == "multi-relation" or self.task_name == "direction-prediction":
            accuracy = accuracy_score(y_true, y_pred)
            aucroc = roc_auc_score(y_true=y_true_multi, y_score=y_prob, multi_class='ovr', average='macro')
            auprc = average_precision_score(y_true=y_true_multi, y_score=y_prob, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            cm = confusion_matrix(y_true, y_pred)
            metrics = {'accuracy': accuracy, 'aucroc': aucroc, 'auprc': auprc, 'f1': f1, 'precision': precision, 'recall': recall, 'confusion_matrix': cm}
        else:
            # compute average distance
            y_true = np.array(y_true)
            y_logit = np.array(y_logit)
            mae = np.mean(np.abs(y_true - y_logit))
            mse = np.mean((y_true - y_logit) ** 2)
            metrics = {'MAE': mae, 'MSE': mse}
        # return the average loss
        return np.mean(val_loss), metrics

    def test(self, test_loader, best_model):
        best_model.eval()
        #import pickle
        #pickle.dump((best_model.encoder.U, best_model.encoder.V), open(f"/home/jiali/Poly2Vec/best_model_learned_frequencies.pkl", "wb"))
        y_true, y_prob, y_pred, y_logit = [], [], [], []
        y_true_multi = []
        with torch.no_grad():
            for batch in test_loader:
                # get batch
                x1 = batch[0].to(self.device)
                x2 = batch[1].to(self.device)
                len1 = batch[2].to(self.device)
                len2 = batch[3].to(self.device)
                y = batch[4].to(self.device)
                y_true.extend(y.cpu().numpy())
                logits = self.model(x1, len1, self.args.dataset_type1, x2, len2, self.args.dataset_type2)

                if self.args.task == "multi-relation" or self.args.task == "direction-prediction":
                    y = y.type(torch.LongTensor).to(self.device)
                    y_true_multi.extend(label_binarize(y.cpu().numpy(), classes=[i for i in range(self.args.num_classes)]))
                out_probs = logits.cpu().numpy()
                # compute metrics
                if self.args.task == "multi-relation" or self.args.task == "direction-prediction":
                    y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().astype(int))
                else:
                    y_logit.extend(logits.cpu().numpy())
                    y_prob.extend(out_probs)
                    y_pred.extend((out_probs > self.best_threshold).astype(int))
        # compute metrics
        if self.task_name == 'relation-prediction':
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='binary')
            precision = precision_score(y_true, y_pred, average='binary')
            recall = recall_score(y_true, y_pred, average='binary')
            aucroc = roc_auc_score(y_true=y_true, y_score=y_prob)
            auprc = average_precision_score(y_true=y_true, y_score=y_prob)
            cm = confusion_matrix(y_true, y_pred)
            metrics = {'accuracy': accuracy, 'f1': f1, 'aucroc': aucroc,
                       'auprc': auprc, 'precision': precision, 'recall': recall, 'confusion_matrix': cm}
        elif self.task_name == "multi-relation" or self.task_name == "direction-prediction":
            accuracy = accuracy_score(y_true, y_pred)
            aucroc = roc_auc_score(y_true=y_true_multi, y_score=y_prob, multi_class='ovr', average='macro')
            auprc = average_precision_score(y_true=y_true_multi, y_score=y_prob, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            cm = confusion_matrix(y_true, y_pred)
            metrics = {'accuracy': accuracy, 'aucroc': aucroc, 'auprc': auprc, 'f1': f1, 'precision': precision, 'recall': recall, 'confusion_matrix': cm}
        else:
            # compute average distance
            y_true = np.array(y_true)
            y_logit = np.array(y_logit)
            mae = np.mean(np.abs(y_true - y_logit))
            mse = np.mean((y_true - y_logit) ** 2)
            metrics = {'MAE': mae, 'MSE': mse}
        # return the average loss
        return metrics



