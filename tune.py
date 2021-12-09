import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from sklearn.model_selection import train_test_split
from copy import deepcopy
import random

from data import get_data
from models.utils import get_model
from funcs import train, test


# arg parse
parser = argparse.ArgumentParser(description='re-sln training')
parser.add_argument('--dataset_name', type=str, choices=["cifar10", "cifar100", "clothing1m"], help='name of the dataset', required=True)
parser.add_argument('--batch_size', type=int, default=128, help='batch size for sgd', required=True)
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for', required=True)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate of optimizer', required=True)
parser.add_argument('--noise_mode', type=str, choices=['sym', 'asym', 'openset', 'dependent'], help='noise mode', required=True)
parser.add_argument('--p', type=float, default=0.4, help='noise rate', required=True)
parser.add_argument('--custom_noise', dest='custom_noise', action='store_true', default=False, help='whether to use custom noise',)
parser.add_argument('--make_new_custom_noise', dest='make_new_custom_noise', action='store_true', default=False, help='whether to generate new custom noise')
parser.add_argument('--mo', dest='mo', action='store_true', default=False, help='whether to use momentum model')
parser.add_argument('--lc_n_epoch', type=int, default=250, help='label correction starts at this epoch (if -1, no lc)', required=True)
parser.add_argument('--val_size', type=float, default=0.1, help='validation split size as float (0.1 is 5k for cifar10 and cifar100)', required=True)
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0, experiments done with 123)', required=True)

datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def hp(config, checkpoint_dir, datapath, dataset_name, noise_mode, p, custom_noise, make_new_custom_noise, seed, batch_size, n_epochs, lr, 
             mo, lc_n_epoch, val_size):

    train_dataset, _, indices_noisy, noise_rules, test_dataset = get_data(
        dataset_name=dataset_name,
        datapath=datapath,
        noise_mode=noise_mode, 
        p=p,
        custom_noise=custom_noise,
        make_new_custom_noise=make_new_custom_noise,
        seed=seed
    )

    # get stratified split (5k noisy samples, so val_size=0.1)
    val_dataset = deepcopy(train_dataset)
    X_train, X_val, y_train, y_val = train_test_split(train_dataset.data, train_dataset.targets, test_size=val_size, stratify=train_dataset.targets,
                                                      random_state=seed)
    train_dataset.data, train_dataset.targets = X_train, y_train
    val_dataset.data, val_dataset.targets = X_val, y_val

    # get number of classes
    n_classes = len(list(train_dataset.class_to_idx.keys()))
    # make targets one-hot (easier to handle in lc and sln), targets_one_hot used in lc
    targets = train_dataset.targets
    targets_one_hot, train_dataset.targets = np.eye(n_classes)[targets], np.eye(n_classes)[targets]
    targets_val = val_dataset.targets
    val_dataset.targets = np.eye(n_classes)[targets_val]
    # train_dataloader is modified if lc is used
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # train_eval_dataloader is never modified, and is used to compute the loss weights for lc
    train_eval_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # val_dataloader
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # get models for naive and ema (depends on dataset)
    model_name = "wrn-28-2" if dataset_name in ["cifar10", "cifar100"] else "MODEL_NAME_FOR_CLOTHING1M"
    model = get_model(model_name=model_name, n_classes=n_classes, device=device)
    # if multi gpu
    if device == "cuda":
        if 1 < torch.cuda.device_count():
            model = torch.nn.DataParallel(model)
    model.to(device)
    # optimizer for model
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # ema model (MO)
    model_ema = get_model(model_name=model_name, n_classes=n_classes, device=device) if mo else None
    if model_ema:
        # no grads for model_ema
        for param in model_ema.parameters():
            param.detach_()
        # if multi gpu
        if device == "cuda":
            if 1 < torch.cuda.device_count():
                model_ema = torch.nn.DataParallel(model_ema)
        model_ema.to(device)
        # ema model optimizer
        optimizer_ema = WeightEMA(model, model_ema, alpha=0.999)
    else:
        optimizer_ema = None
    
    sigma = deepcopy(config["sigma"])
    
    # start experiment
    for n_epoch in range(1, n_epochs+1):
        # label-correction
        # if SLN-MO-LC model
        if model_ema and 0 < lc_n_epoch and lc_n_epoch <= n_epoch:
            # set sigma to 0, no more stochastic label noise as lc starts
            sigma = 0
            # keep targets one hot through lc
            losses, softmaxes = \
                get_lc_params(model_ema=model_ema, train_eval_dataloader=train_eval_dataloader, device=device, 
                              n_epoch=n_epoch, n_epochs=n_epochs, verbose=False)
            # normalize to [0.0, 1.0]
            weights = torch.reshape((losses - torch.min(losses)) / (torch.max(losses) - torch.min(losses)), (len(train_dataloader.dataset), 1))
            weights = weights.numpy()
            preds = np.argmax(softmaxes.numpy(), axis=1).tolist()
            preds_one_hot = np.eye(n_classes)[preds]
            # do lc and reload training data (targets_one_hot fixed variable from above)
            targets_one_hot_lc = weights*targets_one_hot + (1-weights)*preds_one_hot
            train_dataset.targets = targets_one_hot_lc
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        # train
        loss_epoch, accuracy_epoch, loss_noisy_epoch, loss_clean_epoch = train(
            model=model, 
            device=device,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            optimizer_ema=optimizer_ema,
            sigma=sigma,
            n_classes=n_classes,
            n_epoch=n_epoch,
            n_epochs=n_epochs, 
            indices_noisy=indices_noisy,
            verbose=False
        )
        
        # if SLN-MO or SLN-MO-LC model, test with EMA model
        if optimizer_ema:
            loss_val, accuracy_val = test(
                model=model_ema, 
                device=device,
                test_dataloader=val_dataloader,
                n_epoch=n_epoch,
                n_epochs=n_epochs,
                verbose=False)

        # if CE or SLN model, test with model
        else:
            loss_val, accuracy_val = test(
                model=model, 
                device=device,
                test_dataloader=val_dataloader,
                n_epoch=n_epoch,
                n_epochs=n_epochs,
                verbose=False)
        
        tune.report(
            loss_train=loss_epoch,
            loss_train_noisy=loss_noisy_epoch,
            loss_train_clean=loss_clean_epoch,
            accuracy_train=accuracy_epoch, 
            loss_val=loss_val, 
            accuracy_val=accuracy_val)
        

def hptune(sigmas, use_n_cpus_per_trial, use_n_gpus_per_trial, datapath, dataset_name, noise_mode, p, custom_noise, 
           make_new_custom_noise, seed, batch_size, n_epochs, lr, mo, lc_n_epoch, val_size):
    # hp
    config = {"sigma": tune.grid_search(sigmas)}

    reporter = CLIReporter(
        # parameter_columns=["sigma"],
        #metric_columns=["loss", "accuracy", "training_iteration"])
        metric_columns=["loss_train", "loss_train_noisy", "loss_train_clean", 
                        "accuracy_train", "loss_val", "accuracy_val", 
                        "training_iteration"])

    result = tune.run(
        tune.with_parameters(hp, 
                             checkpoint_dir=None, datapath=datapath, dataset_name=dataset_name, 
                             noise_mode=noise_mode, p=p, custom_noise=custom_noise,
                             make_new_custom_noise=make_new_custom_noise,
                             seed=seed, batch_size=batch_size, n_epochs=n_epochs, 
                             lr=lr, mo=mo, lc_n_epoch=lc_n_epoch, val_size=val_size),
        resources_per_trial={"cpu": use_n_cpus_per_trial, "gpu": use_n_gpus_per_trial},
        config=config,
        num_samples=1, #so grid_search is repeated once only
        local_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), "hp"),
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("accuracy_val", "max", "last")
    print(f"best trial config: {best_trial.config}")
    print(f"best trial final validation accuracy: {best_trial.last_result['accuracy_val']}")
    print(f"best trial final validation loss: {best_trial.last_result['loss_val']}")
    
    print(f"best trial final train accuracy: {best_trial.last_result['accuracy_train']}")
    print(f"best trial final train loss: {best_trial.last_result['loss_train']}")
    print(f"best trial final train loss clean: {best_trial.last_result['loss_train_clean']}")
    print(f"best trial final train loss noisy: {best_trial.last_result['loss_train_noisy']}")
        

if __name__ == "__main__":
    # args parse
    args = parser.parse_args()
    
    assert os.cpu_count() == 4 and torch.cuda.device_count() == 2, f"this script needs to be changed if not used with 4 cpus and 2 gpus"
    
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')
    if device == "cuda":
        print(f"using {torch.cuda.device_count()} GPU(s)")

    # reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    use_n_cpus_per_trial = 2
    use_n_gpus_per_trial = 1
    sigmas = [0.1, 0.2, 0.5, 1.0]
    
    hptune(
        sigmas=sigmas,
        use_n_cpus_per_trial=use_n_cpus_per_trial,
        use_n_gpus_per_trial=use_n_gpus_per_trial,
        datapath=datapath, 
        dataset_name=args.dataset_name, 
        noise_mode=args.noise_mode, 
        p=args.p, 
        custom_noise=args.custom_noise,
        make_new_custom_noise=args.make_new_custom_noise,
        seed=args.seed, 
        batch_size=args.batch_size, 
        n_epochs=args.n_epochs, 
        lr=args.lr, 
        mo=args.mo, 
        lc_n_epoch=args.lc_n_epoch, 
        val_size=args.val_size)
