import torch
import json, os, argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from loaders.dataloader import GeometryRelationshipDataset
from utils.utils import CheckpointSaver, get_save_dir, load_model_checkpoint
from utils.config import args_parser
from models.TaskEncoder import TaskEncoder
from models.GeometryEncoder import GeometryEncoder
from utils.trainer import Trainer

VERBOSE = True

def parse_args():
    # Set some key configs from screen

    parser = argparse.ArgumentParser(description="run.py")

    parser.add_argument("-dataset")
    parser.add_argument("-dataset_name")
    parser.add_argument("-dataset_type1")
    parser.add_argument("-dataset_type2")
    parser.add_argument("-task")
    parser.add_argument("-data_file")
    parser.add_argument("-encoder_type")
    parser.add_argument("-data_path")
    parser.add_argument("-num_classes", type=int, default=2)
    parser.add_argument("-sampling_strategy", type=str, default="gfm")
    args = parser.parse_args()

    params = {}
    for param, value in args._get_kwargs():
        if value is not None:
            params[param] = value

    return params


if __name__ == "__main__":
    args = args_parser("config.json")
    
    task_args = parse_args()
    for param, value in task_args.items():
        if hasattr(args, param):
            setattr(args, param, value)

    run_name = f'[{args.encoder_type}]: {args.dataset}_sampling strategy: {args.sampling_strategy}_FT fusion: {args.fusion}_task: {args.task}'
    print(run_name)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print("using device:", device)

    args.save_dir = get_save_dir(f"{args.save_dir}/{run_name}", training=args.training)

    # create a Poly2Vec model
    geometry_model = GeometryEncoder(args, device).to(device)
    model = TaskEncoder(args, geometry_model, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.task == "distance-prediction":
        criterion = torch.nn.MSELoss().to(device)
    elif args.task == "relation-prediction":
        criterion = torch.nn.BCELoss().to(device)
    elif args.task == "multi-relation" or args.task == "direction-prediction":
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        print(f"Unknown task {args.task}!")

    # Train the model first
    # Save args
    args_file = os.path.join(args.save_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    checkpoint_saver = CheckpointSaver(
        args.save_dir, metric_name="eval_loss", maximize_metric=False
    )

    # read dataset
    X1, X2, L1, L2, Y = torch.load(args.data_path + args.data_file)
    # split dataset
    X1_train, X1_other, X2_train, X2_other, L1_train, L1_other, L2_train, L2_other, Y_train, Y_other = train_test_split(X1, X2, L1, L2, Y, test_size=0.4)
    X1_val, X1_test, X2_val, X2_test, L1_val, L1_test, L2_val, L2_test, Y_val, Y_test = train_test_split(X1_other, X2_other, L1_other, L2_other, Y_other, test_size=0.5)

    # batchify
    train_dataset = GeometryRelationshipDataset(X1_train, X2_train, L1_train, L2_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = GeometryRelationshipDataset(X1_val, X2_val, L1_val, L2_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = GeometryRelationshipDataset(X1_test, X2_test, L1_test, L2_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = Trainer(model,
                      optimizer,
                      criterion,
                      train_loader,
                      val_loader,
                      device,
                      args=args,
                      checkpoint_saver=checkpoint_saver, )

    if args.training:
        trainer.run(epochs=args.epochs, verbose=VERBOSE, patience=args.patience)

    # Load the best model
    best_path = os.path.join(args.save_dir, 'best.pth.tar')
    best_model = load_model_checkpoint(best_path, model)

    # Test the model
    test_metrics = trainer.test(test_loader, best_model)

    print(f"Test Metrics: {test_metrics}")
