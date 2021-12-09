import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import random
import datetime
import time
import os

from models.ema import WeightEMA
from models.utils import get_model
from data import get_data
from funcs import train, test, evaluate, get_lc_params
from utils import save_config

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
parser.add_argument('--sigma', type=float, help='std of Gaussian noise of optimizer', required=True)
parser.add_argument('--mo', dest='mo', action='store_true', default=False, help='whether to use momentum model')
parser.add_argument('--lc_n_epoch', type=int, default=250, help='label correction starts at this epoch (if -1, no lc)', required=True)
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0, experiments done with 123)', required=True)

datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

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
    
    # data
    print("preparing data")
    train_dataset, _, indices_noisy, noise_rules, test_dataset = get_data(
        dataset_name=args.dataset_name,
        datapath=datapath,
        noise_mode=args.noise_mode, 
        p=args.p,
        custom_noise=args.custom_noise,
        make_new_custom_noise=args.make_new_custom_noise,
        seed=args.seed
    )
    # get number of classes
    n_classes = len(list(train_dataset.class_to_idx.keys()))
    # make targets one-hot (easier to handle in lc and sln), targets_one_hot used in lc
    targets = train_dataset.targets
    targets_one_hot, train_dataset.targets = np.eye(n_classes)[targets], np.eye(n_classes)[targets]
    targets_test = test_dataset.targets
    test_dataset.targets = np.eye(n_classes)[targets_test]
    # train_dataloader is modified if lc is used
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # train_eval_dataloader is never modified, and is used to compute the loss weights for lc
    train_eval_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # test_dataloader is never modified (test dataset is not onehot yet?)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # get models for naive and ema (depends on dataset)
    print("loading models")
    model_name = "wrn-28-2" if args.dataset_name in ["cifar10", "cifar100"] else "MODEL_NAME_FOR_CLOTHING1M"
    model = get_model(model_name=model_name, n_classes=n_classes, device=device)
    # if multi gpu
    if device == "cuda":
        if 1 < torch.cuda.device_count():
            model = torch.nn.DataParallel(model)
    model.to(device)
    # optimizer for model
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # ema model (MO)
    model_ema = get_model(model_name=model_name, n_classes=n_classes, device=device) if args.mo else None
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
    
    # logging and tensorboard stuff
    datetime_now = datetime.datetime.now()
    exp_id = f"exp_{datetime_now}"
    writer = SummaryWriter(f"runs/{exp_id}/")
    
    # experiment reporting
    ce_cond = args.sigma == 0 and model_ema is None and args.lc_n_epoch == -1
    sln_cond = 0 < args.sigma and model_ema is None and args.lc_n_epoch == -1
    sln_mo_cond = 0 < args.sigma and model_ema and args.lc_n_epoch == -1
    sln_mo_lc_cond = 0 < args.sigma and model_ema and 0 < args.lc_n_epoch < args.n_epochs
    assert ce_cond or sln_cond or sln_mo_cond or sln_mo_lc_cond, "incorrect experiment, check arguemnts: sigma, mo, and lc_n_epoch"
    exp_str = \
        f'ce' if ce_cond else \
        f'sln (sigma={args.sigma})' if sln_cond else \
        f'sln_mo (sigma={args.sigma})' if sln_mo_cond else \
        f'sln_mo_lc (sigma={args.sigma}, lc={args.lc_n_epoch})' if sln_mo_lc_cond else \
        None
    print(f"exp_id: {exp_id}\n{args.dataset_name}, {args.noise_mode} (p={args.p}), {'custom noise' if args.custom_noise else 'paper noise'}\n{exp_str}")
    
    # save lossess and accuracies in lists
    loss_epochs = []
    loss_noisy_epochs = []
    loss_clean_epochs = []
    accuracy_epochs = []
    loss_test_epochs = []
    accuracy_test_epochs = []
    
        
    # save config
    save_config(
        exp_id=exp_id,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        noise_mode=args.noise_mode,
        p=args.p,
        custom_noise=args.custom_noise,
        make_new_custom_noise=args.make_new_custom_noise, 
        sigma=args.sigma,
        mo=args.mo,
        lc_n_epoch=args.lc_n_epoch,
        seed=args.seed
    )
    
    # start experiment
    for n_epoch in range(1, args.n_epochs+1):
        # label-correction
        # if SLN-MO-LC model
        if model_ema and 0 < args.lc_n_epoch and args.lc_n_epoch <= n_epoch:
            # set sigma to 0, no more stochastic label noise as lc starts
            args.sigma = 0
            # keep targets one hot through lc
            losses, softmaxes = \
                get_lc_params(model_ema=model_ema, train_eval_dataloader=train_eval_dataloader, device=device, n_epoch=n_epoch, n_epochs=args.n_epochs)
            # normalize to [0.0, 1.0]
            weights = torch.reshape((losses - torch.min(losses)) / (torch.max(losses) - torch.min(losses)), (len(train_dataloader.dataset), 1))
            weights = weights.numpy()
            preds = np.argmax(softmaxes.numpy(), axis=1).tolist()
            preds_one_hot = np.eye(n_classes)[preds]
            # do lc and reload training data (targets_one_hot fixed variable from above)
            targets_one_hot_lc = weights*targets_one_hot + (1-weights)*preds_one_hot
            train_dataset.targets = targets_one_hot_lc
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        # train
        loss_epoch, accuracy_epoch, loss_noisy_epoch, loss_clean_epoch = train(
            model=model, 
            device=device,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            optimizer_ema=optimizer_ema,
            sigma=args.sigma,
            n_classes=n_classes,
            n_epoch=n_epoch,
            n_epochs=args.n_epochs, 
            indices_noisy=indices_noisy
        )

        # tensorboard stuff
        writer.add_scalar("accuracy/train", accuracy_epoch, n_epoch)
        writer.add_scalars('loss/train', {'all': loss_epoch,
                                          'noisy': loss_noisy_epoch,
                                          'clean': loss_clean_epoch}, n_epoch)
        # append to lists
        loss_epochs.append(loss_epoch)
        loss_noisy_epochs.append(loss_noisy_epoch)
        loss_clean_epochs.append(loss_clean_epoch)
        accuracy_epochs.append(accuracy_epoch)
        
        # if SLN-MO or SLN-MO-LC model, test with EMA model
        if optimizer_ema:
            loss_test, accuracy_test = test(
                model=model_ema, 
                device=device,
                test_dataloader=test_dataloader,
                n_epoch=n_epoch,
                n_epochs=args.n_epochs)

            writer.add_scalar("loss/test", loss_test, n_epoch)
            writer.add_scalar("accuracy/test", accuracy_test, n_epoch)
            loss_test_epochs.append(loss_test)
            accuracy_test_epochs.append(accuracy_test)

            print(f"epoch={n_epoch}/{args.n_epochs}, loss_epoch={loss_epoch:.4f}, acc_epoch={accuracy_epoch:.4f}, "
                  f"loss_test={loss_test:.4f}, accuracy_test={accuracy_test:.4f}")
        # if CE or SLN model, test with model
        else:
            loss_test, accuracy_test = test(
                model=model, 
                device=device,
                test_dataloader=test_dataloader,
                n_epoch=n_epoch,
                n_epochs=args.n_epochs)

            writer.add_scalar("loss/test", loss_test, n_epoch)
            writer.add_scalar("accuracy/test", accuracy_test, n_epoch)
            loss_test_epochs.append(loss_test)
            accuracy_test_epochs.append(accuracy_test)

            print(f"epoch={n_epoch}/{args.n_epochs}, loss_epoch={loss_epoch:.4f}, acc_epoch={accuracy_epoch:.4f}, "
                  f"loss_test={loss_test:.4f}, accuracy_test={accuracy_test:.4f}")

    # Call flush() method to make sure that all pending events have been written to disk
    writer.flush()
    
    # Saving model (and ema_model if exists) 
    model_save_path = Path(f"saved_models/{exp_id}/")
    model_save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path / "model.pth")
    if model_ema:
        torch.save(model_ema.state_dict(), model_save_path / "model_ema.pth")
    print(f"all models saved, experiment exp_id: {exp_id} completed")
