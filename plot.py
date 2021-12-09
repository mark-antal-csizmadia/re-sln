import random
import argparse
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import numpy as np
from models.utils import load_model
from utils import load_config
from data import get_data
from funcs import evaluate


parser = argparse.ArgumentParser(description='re-sln plotting')
parser.add_argument('--exp_id', type=str, help='experiment id', required=True)
parser.add_argument('--plot_type', type=str, choices=["pred_probs", "sample_dissect", "tsne"], help='what plot to produce', required=True)

datapath = "data/"



def plot_prediction_probabilities(softmaxes, indices_noisy, title, viz_save_path=None):
    viz_save_path.mkdir(parents=True, exist_ok=True)
    
    softmaxes_max_object = torch.max(softmaxes, dim=1)
    probs, preds = softmaxes_max_object[0], softmaxes_max_object[1]
    
    probs_noisy = probs[indices_noisy]
    probs_clean = probs[np.invert(indices_noisy)]
    
    fig = plt.figure(figsize=(4,4))
    
    # 100 bins
    bins = np.linspace(0, 10, 100) / 10

    plt.hist(probs_clean.numpy(), bins, alpha=0.5, color="b", label='Clean')
    plt.hist(probs_noisy.numpy(), bins, alpha=0.5, color="r", label='Noisy')
    
    plt.legend(loc='best')
    
    plt.xticks([i/100 for i in range(0,101,25)])
    plt.grid("on")
    plt.title(title)
    plt.xlabel("Prediction probability")
    plt.ylabel("Number of samples")
    
    # y label is off when saved
    plt.tight_layout()
    
    if viz_save_path:
        viz_save_path = viz_save_path / f"{title}-prediction-probabilities.png"
        plt.savefig(viz_save_path)
        print(f"generated: {viz_save_path}")
    
    plt.show()
    
    
def plot_sample_dissection(train_eval_dataloader, train_original_dataloader, predictions, losses, indices_noisy, title, viz_save_path):
    viz_save_path.mkdir(parents=True, exist_ok=True)
    
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/stairs_demo.html
    n_data = len(train_eval_dataloader.dataset)
    
    # noisy targets and clean (original targets)
    noisy_targets = torch.argmax(torch.tensor(train_eval_dataloader.dataset.targets), dim=1)
    clean_targets = torch.tensor(train_original_dataloader.dataset.targets)
    
    # if noisy and if correct target prediciton
    correct_labels = clean_targets == noisy_targets
    correct_predictions = clean_targets == predictions
    
    # types:
    # 1: label x, prediction x
    # 2: label x, prediction y
    # 3: label y, prediction x
    # 4: label y, prediction y
    
    type_1 = torch.logical_not(torch.logical_or(correct_labels, correct_predictions))
    type_2 = torch.logical_and(torch.logical_not(correct_labels), correct_predictions)
    type_3 = torch.logical_and(correct_labels, torch.logical_not(correct_predictions))
    type_4 = torch.logical_and(correct_labels, correct_predictions)
    types = torch.ones((n_data,), dtype=int)
    types[type_2] = 2
    types[type_3] = 3
    types[type_4] = 4
    
    # sort loss in ascending order
    losses_sorted, indices = torch.sort(losses, descending=False)
    
    # divide the indices into 50 bins of 1000 samples
    n_samples_per_bins = 1000
    n_bins = int(n_data / n_samples_per_bins)
    indices_split = torch.split(indices, split_size_or_sections=n_samples_per_bins)
    
    # get the number of type_xs in the 50 bins
    type_1_hist = []
    type_2_hist = []
    type_3_hist = []
    type_4_hist = []
    
    for i in range(len(indices_split)):
        type_1_hist.append((types[indices_split[i]]==1).sum().item())
        type_2_hist.append((types[indices_split[i]]==2).sum().item())
        type_3_hist.append((types[indices_split[i]]==3).sum().item())
        type_4_hist.append((types[indices_split[i]]==4).sum().item())
    
    type_1_hist = np.array(type_1_hist)
    type_2_hist = np.array(type_2_hist)
    type_3_hist = np.array(type_3_hist)
    type_4_hist = np.array(type_4_hist)
    
    # plot a stair, and add to the ones in the backgorund
    type_2_hist = type_1_hist + type_2_hist
    type_3_hist = type_2_hist + type_3_hist
    type_4_hist = type_3_hist + type_4_hist
    
    fig = plt.figure(figsize=(4,4))
    
    # make stairs plot
    # 1: label x, prediction x
    # 2: label x, prediction y
    # 3: label y, prediction x
    # 4: label y, prediction y
    plt.stairs(type_4_hist, fill=True, color="r", label='$l: \checkmark, p: \checkmark$')
    plt.stairs(type_3_hist, fill=True, color="g", label='$l: \checkmark, p: x$')
    plt.stairs(type_2_hist, fill=True, color="y", label='$l: x, p: \checkmark$')
    plt.stairs(type_1_hist, fill=True, color="b", label='$l: x, p: x$')
    
    # set limits to delete margins around the stair plot
    plt.xlim(0, n_bins)
    plt.ylim(0, n_samples_per_bins)
    
    # plt stuff
    plt.title(title)
    plt.xlabel(r"Intervals (loss $\uparrow$)")
    plt.ylabel("Number of samples")
    plt.legend(loc="upper left")
    
    # y label is off when saved
    plt.tight_layout()
    
    if viz_save_path:
        viz_save_path = viz_save_path / f"{title}-sample-dissection.png"
        plt.savefig(viz_save_path)
        print(f"generated: {viz_save_path}")
    
    plt.show()

    
if __name__ == "__main__":
    # args parse
    args = parser.parse_args()
    
    # load config of exp_id
    try:
        config_data = load_config(exp_id=args.exp_id)
    except:
        raise FileNotFoundError(f"{args.exp_id} does not seem to be a saved experiment")
    
    # cuda stuff
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')
    if device == "cuda":
        print(f"using {torch.cuda.device_count()} GPU(s)")

    # reproducibility
    random.seed(config_data["seed"])
    torch.manual_seed(config_data["seed"])
    torch.cuda.manual_seed_all(config_data["seed"])
    
    # data
    print("preparing data")
    train_dataset, train_dataset_original, indices_noisy, noise_rules, test_dataset = get_data(
        dataset_name=config_data["dataset_name"],
        datapath=datapath,
        noise_mode=config_data["noise_mode"], 
        p=config_data["p"],
        custom_noise=config_data["custom_noise"],
        # never make new custom noise here
        make_new_custom_noise=False,
        seed=config_data["seed"]
    )
    # get number of classes
    n_classes = len(list(train_dataset.class_to_idx.keys()))
    # make targets one-hot (easier to handle in lc and sln), targets_one_hot used in lc
    targets = train_dataset.targets
    targets_one_hot, train_dataset.targets = np.eye(n_classes)[targets], np.eye(n_classes)[targets]
    targets_test = test_dataset.targets
    test_dataset.targets = np.eye(n_classes)[targets_test]
    # train_dataloader is modified if lc is used
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config_data["batch_size"], shuffle=True, num_workers=2)
    # fully clean, for viz only
    train_original_dataloader = torch.utils.data.DataLoader(train_dataset_original, batch_size=config_data["batch_size"], shuffle=True, num_workers=2)
    # train_eval_dataloader is never modified, and is used to compute the loss weights for lc
    train_eval_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config_data["batch_size"], shuffle=False, num_workers=2)
    # test_dataloader is never modified (test dataset is not onehot yet?)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config_data["batch_size"], shuffle=False, num_workers=2)
    
    # load model
    model = load_model(exp_id=args.exp_id, dataset_name=config_data["dataset_name"], n_classes=n_classes, device=device)
    
    # plot related things
    viz_save_path = Path(f"assets/{args.exp_id}")
    model_type = \
        "ce" if config_data["sigma"] == 0 and not config_data["mo"] and config_data["lc_n_epoch"] == -1 else \
        "sln" if 0 < config_data["sigma"] and not config_data["mo"] and config_data["lc_n_epoch"] == -1 else \
        "sln-mo" if 0 < config_data["sigma"] and config_data["mo"] and config_data["lc_n_epoch"] == -1 else \
        "sln-mo-lc" if 0 < config_data["sigma"] and config_data["mo"] and config_data["lc_n_epoch"] != -1 else None
    noise_type = f"custom" if config_data['custom_noise'] else "paper"
    title = f"{config_data['dataset_name']}-{model_type}-{config_data['noise_mode']}-{noise_type}-{config_data['p']}"
    
    if args.plot_type == "pred_probs":
        loss, accuracy, losses, softmaxes, predictions = evaluate(model=model, device=device, dataloader=train_eval_dataloader, verbose=True)
        plot_prediction_probabilities(softmaxes=softmaxes, indices_noisy=indices_noisy, title=title, viz_save_path=viz_save_path)
        
    elif args.plot_type == "sample_dissect":
        loss, accuracy, losses, softmaxes, predictions = evaluate(model=model, device=device, dataloader=train_eval_dataloader, verbose=True)
        plot_sample_dissection(
            train_eval_dataloader=train_eval_dataloader, 
            train_original_dataloader=train_original_dataloader, 
            predictions=predictions, 
            losses=losses,
            indices_noisy=indices_noisy,
            title=title,
            viz_save_path=viz_save_path
        )
