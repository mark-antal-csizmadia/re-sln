import random
import argparse
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import os

from data import get_data, get_data_real
from data import Animal10N


parser = argparse.ArgumentParser(description='re-sln viz')

base_parser = argparse.ArgumentParser(add_help=False)
subparsers = parser.add_subparsers(dest='dataset_name', help='name of the dataset')

parser_cifar10 = subparsers.add_parser('cifar10', help='cifar10', parents=[base_parser])
parser_cifar10.add_argument('--noise_mode', type=str, choices=['sym', 'asym', 'openset', 'dependent'], help='noise mode', required=True)
parser_cifar10.add_argument('--p', type=float, default=0.4, help='noise rate', required=True)
parser_cifar10.add_argument('--custom_noise', dest='custom_noise', action='store_true', default=False, help='whether to use custom noise',)
parser_cifar10.add_argument('--seed', type=int, default=0, help='random seed (default: 0, experiments done with 123)', required=True)

parser_cifar100 = subparsers.add_parser('cifar100', help='cifar100', parents=[base_parser])
parser_cifar100.add_argument('--noise_mode', type=str, choices=['sym', 'asym', 'openset', 'dependent'], help='noise mode', required=True)
parser_cifar100.add_argument('--p', type=float, default=0.4, help='noise rate', required=True)
parser_cifar100.add_argument('--custom_noise', dest='custom_noise', action='store_true', default=False, help='whether to use custom noise',)
parser_cifar100.add_argument('--seed', type=int, default=0, help='random seed (default: 0, experiments done with 123)', required=True)

parser_animal10n = subparsers.add_parser('animal-10n', help='animal-10n', parents=[base_parser])
parser_animal10n.add_argument('--seed', type=int, default=0, help='random seed (default: 0, experiments done with 123)', required=True)

datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def show_imgs(datapath, dataset_name, train_dataset, noise_mode, indices_noisy, save_path, seed):
    """ Show image from a noisy dataset. 
    
    Parameters
    ----------
    datapath : str
        Path to downloaded data sets.
    dataset_name : str
        Name of the dataset.
    train_dataset : torchvision.datasets.x.x
        The training dataset split, e.g.: torchvision.datasets.cifar.CIFAR10
    noise_mode : str
        sym, asym, openset, dependent
    indices_noisy : list
        Boolean list of Trues (noisy) and Falses (clean)
    save_path : str
        Save fig path.
    seed : int
        Seed for reproducibility.
    
    Returns
    -------
    None
    """
    # get the clean class labels in the taining dataset
    class_to_idx_list=list(train_dataset.class_to_idx.keys())
    
    # load the same training dataset for visualization pruposes
    if dataset_name == "cifar10":
        train_dataset_viz = datasets.CIFAR10(os.path.join(datapath, dataset_name), train=True, transform=transforms.ToTensor(), download=True)
    elif dataset_name == "cifar100":
        train_dataset_viz = datasets.CIFAR100(os.path.join(datapath, dataset_name), train=True, transform=transforms.ToTensor(), download=True)
    
    # get the clean targets and the clean data (not noisy) for viz purposes
    targets_clean = train_dataset_viz.targets
    train_dataset_viz.targets = train_dataset.targets
    train_dataset_viz.data = train_dataset.data
    
    # get the indices of noisy and clean instances
    indices_noisy_args = np.argwhere(indices_noisy).flatten()
    indices_clean_args = np.argwhere(indices_noisy == False).flatten()
    
    # plt stuff
    figure = plt.figure(figsize=(8*3, 8))
    cols, rows = 4*2, 4
    half = int(cols * rows / 2)
    
    # plot images
    for i in range(1, cols * rows + 1):
        # the sample indices for images with clean and noisy labels
        # noisy
        if half < i:
            np.random.seed(seed+i)
            sample_idx = np.random.choice(indices_noisy_args, 1, replace=False)[0]
        # clean
        else:
            np.random.seed(seed+i)
            sample_idx = np.random.choice(indices_clean_args, 1, replace=False)[0]
        
        # get clean image and label for sample indices
        img, label = train_dataset_viz[sample_idx]
        figure.add_subplot(rows, cols, i)
        
        # if noisy, add clean label too
        if half < i:
            # sym, asym, and dependet are class label flips
            if noise_mode in ["sym", "asym", "dependent"]:
                plt.title(f"dirty:{class_to_idx_list[label]} ({class_to_idx_list[targets_clean[sample_idx]]})")
            # openset is image flip so label is same as in original dataset but image flipped to out-of-dist images
            elif noise_mode  == "openset":
                plt.title(f"dirty:({class_to_idx_list[label]})")
        # clean
        else:
            plt.title(f"clean:{class_to_idx_list[label]}")
            
        plt.axis("off")
        # pytorch has dimension differently from what plt expects so permute first
        plt.imshow(img.permute(1, 2, 0))
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()
    

def show_imgs_real(datapath, dataset_name, train_dataset, save_path, seed):
    """ Show image from a real-world noisy dataset. 
    
    Returns
    -------
    None
    """
    # get the clean class labels in the taining dataset
    class_to_idx_list=list(train_dataset.class_to_idx.keys())
    
    # load the same training dataset for visualization pruposes
    if dataset_name == "animal-10n":
        train_dataset_viz = Animal10N(root=os.path.join(datapath, dataset_name), train=True, transform=transforms.ToTensor())
    else:
        raise Exception
    
    # plt stuff
    figure = plt.figure(figsize=(8*3, 8))
    cols, rows = 4*2, 4
    half = int(cols * rows / 2)
    
    # plot images
    for i in range(1, cols * rows + 1):
        np.random.seed(seed+i)
        sample_idx = np.random.choice(np.arange(train_dataset_viz.data.shape[0]), 1, replace=False)[0]
        
        img, label, _ = train_dataset_viz[sample_idx]
        figure.add_subplot(rows, cols, i)
        
        plt.title(f"{class_to_idx_list[label]}")
            
        plt.axis("off")
        # pytorch has dimension differently from what plt expects so permute first
        plt.imshow(img.permute(1, 2, 0))
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()

    
if __name__ == "__main__":
    # args parse
    args = parser.parse_args()
    
    # cuda stuff
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')
    if device == "cuda":
        print(f"using {torch.cuda.device_count()} GPU(s)")

    # reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if args.dataset_name in ["cifar10", "cifar100"]: 
        # data
        print("preparing data")
        train_dataset, train_dataset_original, indices_noisy, noise_rules, test_dataset = get_data(
            dataset_name=args.dataset_name,
            datapath=datapath,
            noise_mode=args.noise_mode, 
            p=args.p,
            custom_noise=args.custom_noise,
            # never make new custom noise here
            make_new_custom_noise=False,
            seed=args.seed
        )
        
        print(train_dataset)
        print(f"dataset_name:{args.dataset_name}, noise_mode:{args.noise_mode}, "
              f"noise_ratio:{indices_noisy.sum() / len(train_dataset.targets):.4f}")
        print("noise_rules")
        print(noise_rules)
        
        save_path = \
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                         "viz", 
                         f"{args.dataset_name}_{args.noise_mode}_p_{args.p}_{'custom' if args.custom_noise else 'paper'}.png")
        
        
        show_imgs(
            datapath=datapath, 
            dataset_name=args.dataset_name, 
            train_dataset=train_dataset,
            noise_mode=args.noise_mode,
            indices_noisy=indices_noisy,
            save_path=save_path,
            seed=args.seed
        )
    else:
        save_path = \
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "viz", f"{args.dataset_name}.png")
        print("preparing data")
        train_dataset, train_dataset_original, test_dataset = get_data_real(dataset_name=args.dataset_name, datapath=datapath)
        print(train_dataset)
        show_imgs_real(datapath=datapath, dataset_name=args.dataset_name, train_dataset=train_dataset, save_path=save_path, seed=args.seed)
    
    print(f"saved: {save_path}")
