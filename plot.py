import matplotlib.pyplot as plt
import torch
import numpy as np


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
        plt.savefig(viz_save_path / f"{title}_prediction_probabilities.png")
    
    plt.show()
    
    
def plot_sample_dissection(train_eval_dataloader, train_original_dataloader, predictions, losses, indices_noisy, title, viz_save_path):
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
    plt.stairs(type_4_hist, fill=True, color="r", label='lol')
    plt.stairs(type_3_hist, fill=True, color="g", label='lol')
    plt.stairs(type_2_hist, fill=True, color="y", label='lol')
    plt.stairs(type_1_hist, fill=True, color="b", label='lol')
    
    # set limits to delete margins around the stair plot
    plt.xlim(0, n_bins)
    plt.ylim(0, n_samples_per_bins)
    
    # plt stuff
    plt.title(title)
    plt.xlabel(r"Intervals (loss $\uparrow$)")
    plt.ylabel("Number of samples")
    
    # y label is off when saved
    plt.tight_layout()
    
    if viz_save_path:
        plt.savefig(viz_save_path / f"{title}_sample_dissection.png")
    
    plt.show()
