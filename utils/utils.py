import os, queue, shutil
import torch
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np

# CheckpointSaver, load_model_checkpoint, get_save_dir are adopted from https://github.com/tsy935/eeg-gnn-ssl

class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self, save_dir, metric_name, maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        print(
            "Saver will {}imize {}...".format(
                "max" if maximize_metric else "min", metric_name
            )
        )

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return (self.maximize_metric and self.best_val <= metric_val) or (
                not self.maximize_metric and self.best_val >= metric_val
        )

    def save(self, epoch, model, optimizer, metric_val):
        """Save model parameters to disk.
        Args:
            epoch (int): Current epoch.
            model (torch.nn.DataParallel): Model to save.
            optimizer: optimizer
            metric_val (float): Determines whether checkpoint is best so far.
        """
        ckpt_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(self.save_dir, "last.pth.tar")
        torch.save(ckpt_dict, checkpoint_path)

        best_path = ""
        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, "best.pth.tar")
            shutil.copy(checkpoint_path, best_path)
            print("New best checkpoint at epoch {}...".format(epoch))


def load_model_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        return model, optimizer

    return model


def get_save_dir(base_dir, training, id_max=500):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        save_dir = os.path.join(base_dir, "train", "{:02d}".format(uid))
        if not os.path.exists(save_dir):
            if training:
                os.makedirs(save_dir)
                return save_dir
            else:
                return os.path.join(base_dir, "train", "{:02d}".format(uid - 1))


def count_parameters(model):
    """
    Counter total number of parameters, for Pytorch
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def thresh_max_f1(y_true, y_prob):
    """
    Find best threshold based on precision-recall curve to maximize F1-score.
    Binary calssification only
    """
    if len(set(y_true)) > 2:
        raise NotImplementedError

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh_filt = []
    fscore = []
    n_thresh = len(thresholds)
    for idx in range(n_thresh):
        curr_f1 = (2 * precision[idx] * recall[idx]) / \
                  (precision[idx] + recall[idx])
        if not (np.isnan(curr_f1)):
            fscore.append(curr_f1)
            thresh_filt.append(thresholds[idx])
    # locate the index of the largest f score
    ix = np.argmax(np.array(fscore))
    best_thresh = thresh_filt[ix]
    return best_thresh