from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import sys


def train(model, device, train_dataloader, optimizer, optimizer_ema, sigma, n_classes, n_epoch, n_epochs, indices_noisy, verbose=True):
    # train mode for model e.g.: dropout, batch norm etc
    model.train()
    
    # record loss per epoch
    loss_epoch_sum = 0.0
    loss_noisy_epoch_sum = 0.0
    loss_clean_epoch_sum = 0.0
    correct_predictions_epoch_sum = 0
    
    # n instances in training set
    n_batches = len(train_dataloader)
    n_data = len(train_dataloader.dataset)
    n_noisy_all = indices_noisy.sum()
    
    # tqdm
    if verbose:
        train_dataloader_tqdm = tqdm(enumerate(train_dataloader), total=n_batches,
                                     file=sys.stdout, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        train_dataloader_tqdm.set_description(f"epoch={n_epoch}/{n_epochs} staring ...")
    else:
        train_dataloader_tqdm = enumerate(train_dataloader)

    for n_batch, (data_batch, targets_batch_one_hot, idx_batch) in train_dataloader_tqdm:
        # since zero-indexed
        n_batch += 1
        batch_size = data_batch.size(0)
        assert len(targets_batch_one_hot.size()) == 2
        # make list of integer targets from one-hot (accuracy calculations)
        targets_batch = targets_batch_one_hot.argmax(dim=1)
        # put data and targets onto device
        data_batch, targets_batch_one_hot, targets_batch = data_batch.to(device), targets_batch_one_hot.to(device), targets_batch.to(device)
        
        # if SLN/SLN-MO/SLN-MO-LC model
        if 0 < sigma:
            # make and add to targets sln of shape targets_batch_one_hot.size() = (batch_size, n_classes)
            targets_batch_one_hot += sigma*torch.randn(targets_batch_one_hot.size()).to(device)
        else:
            pass
        
        # get model logits
        logits_batch = model(data_batch)
        # get cross entropy (ce) loss, i.e.: negative log-lieklihood
        # use log of softmax for numerical stability and calucalte the cross entropy loss manually
        losses_batch = -torch.sum(F.log_softmax(logits_batch, dim=1)*targets_batch_one_hot, dim=1)
        loss_batch = torch.mean(losses_batch)
        
        n_noisy = indices_noisy[idx_batch.numpy()].sum()
        
        loss_batch_noisy = torch.mean(losses_batch[indices_noisy[idx_batch.numpy()]])
        loss_batch_clean = torch.mean(losses_batch[np.invert(indices_noisy[idx_batch.numpy()])])
        
        # get predictions
        predictions_batch = logits_batch.argmax(dim=1, keepdim=True).to(device)
        # get correct predictions (boolean vector, True if correct), view predictions_batch as targets_batch
        correct_predictions_batch = predictions_batch.view_as(targets_batch).eq(targets_batch)
        correct_predictions_epoch_sum += correct_predictions_batch.sum().item()
        # accuracy of batch
        acc_batch = correct_predictions_batch.sum() / batch_size
        
        # zero out grads, b default their are accumulated over steps
        optimizer.zero_grad()
        # backprop to obtain grads for model params
        loss_batch.backward()
        # apply model params
        optimizer.step()
        # if SLN-MO/SLN-MO-LC model
        if optimizer_ema:
            # no zero grad as custom optimizer, see its class
            optimizer_ema.step()
        
        # accumulate loss per batch, i.e.: add the loss per batch batch_size times
        # eventually mean is computed loss is computed by dividing by the datset size
        loss_epoch_sum += batch_size * loss_batch.item()
        loss_noisy_epoch_sum += n_noisy * loss_batch_noisy.item()
        loss_clean_epoch_sum += (batch_size - n_noisy) * loss_batch_clean.item()
        
        # tqdm
        if verbose:
            train_dataloader_tqdm.set_description(f"epoch={n_epoch}/{n_epochs}, "
                                                  f"batch={n_batch}/{n_batches}, "
                                                  f"loss_batch={loss_batch.item():.4f}, "
                                                  f"acc_batch={acc_batch.item():.4f}")
        else:
            pass

    # compute loss per epoch as the mean of the loss_batches
    loss_epoch = loss_epoch_sum / n_data
    loss_noisy_epoch = loss_noisy_epoch_sum / n_noisy_all
    loss_clean_epoch = loss_clean_epoch_sum / (n_data - n_noisy_all)
    # accuracy epoch
    accuracy_epoch = correct_predictions_epoch_sum / n_data
    
    return loss_epoch, accuracy_epoch, loss_noisy_epoch, loss_clean_epoch

def test(model, device, test_dataloader, n_epoch, n_epochs, verbose=True):
    # eval mode for test
    model.eval()
    
    n_data = len(test_dataloader.dataset)
    
    # record loss per epoch
    loss_sum = 0.0
    correct_predictions_sum = 0
    
    # tqdm
    n_batches = len(test_dataloader)
    if verbose:
        test_dataloader_tqdm = tqdm(enumerate(test_dataloader), total=n_batches,
                                   file=sys.stdout, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        test_dataloader_tqdm.set_description(f"epoch={n_epoch}/{n_epochs}, eval test set ...")
    else:
        test_dataloader_tqdm = enumerate(test_dataloader)
    
    with torch.no_grad():
        for n_batch, (data_batch, targets_batch_one_hot, idx_batch) in test_dataloader_tqdm:
            # since zero-indexed
            n_batch += 1
            batch_size = data_batch.size(0)
            
            assert len(targets_batch_one_hot.size()) == 2
            
            # make list of integer targets from one-hot (accuracy calculations)
            targets_batch = targets_batch_one_hot.argmax(dim=1)
            # put data and targets onto device
            data_batch, targets_batch_one_hot, targets_batch = data_batch.to(device), targets_batch_one_hot.to(device), targets_batch.to(device)
            
            # get model logits
            logits_batch = model(data_batch)
            # get cross entropy (ce) loss, i.e.: negative log-lieklihood
            # use log of softmax for numerical stability and calucalte the cross entropy loss manually
            loss_batch = -torch.mean(torch.sum(F.log_softmax(logits_batch, dim=1)*targets_batch_one_hot, dim=1))
            # get predictions
            predictions_batch = logits_batch.argmax(dim=1, keepdim=True).to(device)
            # get correct predictions (boolean vector, True if correct), view predictions_batch as targets_batch
            correct_predictions_batch = predictions_batch.view_as(targets_batch).eq(targets_batch)
            correct_predictions_sum += correct_predictions_batch.sum().item()
            # accuracy of batch
            acc_batch = correct_predictions_batch.sum() / batch_size
            
            # accumulate loss per batch, i.e.: add the loss per batch batch_size times
            # eventually mean is computed loss is computed by dividing by the datset size
            loss_sum += batch_size * loss_batch.item()
            
            # tqdm
            if verbose:
                test_dataloader_tqdm.set_description(f"epoch={n_epoch}/{n_epochs}, eval test set: "
                                                     f"epoch={n_epoch}/{n_epochs}, "
                                                     f"batch={n_batch}/{n_batches}, "
                                                     f"loss_batch={loss_batch.item():.4f}, "
                                                     f"acc_batch={acc_batch.item():.4f}")
            else:
                pass

    # compute loss test
    loss = loss_sum / n_data
    # accuracy test
    accuracy = correct_predictions_sum / n_data
    
    return loss, accuracy


def evaluate(model, device, dataloader, verbose):
    # eval mode for test
    model.eval()
    
    n_data = len(dataloader.dataset)
    
    # record loss per epoch
    loss_sum = 0.0
    correct_predictions_sum = 0
    
    softmaxes = []
    losses = []
    predictions = []
    
    # tqdm
    n_batches = len(dataloader)
    if verbose:
        dataloader_tqdm = tqdm(enumerate(dataloader), total=n_batches,
                              file=sys.stdout, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        dataloader_tqdm.set_description(f"eval set")
    else:
        pass
    
    with torch.no_grad():
        for n_batch, (data_batch, targets_batch_one_hot, idx) in dataloader_tqdm:
            # since zero-indexed
            n_batch += 1
            batch_size = data_batch.size(0)
            
            assert len(targets_batch_one_hot.size()) == 2
            
            # make list of integer targets from one-hot (accuracy calculations)
            targets_batch = targets_batch_one_hot.argmax(dim=1)
            # put data and targets onto device
            data_batch, targets_batch_one_hot, targets_batch = data_batch.to(device), targets_batch_one_hot.to(device), targets_batch.to(device)
            
            # get model logits
            logits_batch = model(data_batch)
            
            # get softmax from logits
            softmax_batch = F.softmax(logits_batch, dim=1)
            
            # get cross entropy (ce) loss, i.e.: negative log-lieklihood
            # use log of softmax for numerical stability and calucalte the cross entropy loss manually
            losses_batch = -torch.sum(F.log_softmax(logits_batch, dim=1)*targets_batch_one_hot, dim=1)
            loss_batch = torch.mean(losses_batch)
            
            # get predictions
            predictions_batch = logits_batch.argmax(dim=1, keepdim=True).to(device)
            
            # get correct predictions (boolean vector, True if correct), view predictions_batch as targets_batch
            correct_predictions_batch = predictions_batch.view_as(targets_batch).eq(targets_batch)
            correct_predictions_sum += correct_predictions_batch.sum().item()
            # accuracy of batch
            acc_batch = correct_predictions_batch.sum() / batch_size
            
            
            
            #predictions_batch = softmax_batch.argmax(axis=1)
            # If the tensor is on a device other than "cpu", you will need to bring it back to the CPU before you can call the .numpy() method. 
            # append to lists
            losses.append(losses_batch.to("cpu").numpy())
            softmaxes.append(softmax_batch.to("cpu").numpy())
            predictions.append(predictions_batch.to("cpu").numpy())
            
            # accumulate loss per batch, i.e.: add the loss per batch batch_size times
            # eventually mean is computed loss is computed by dividing by the datset size
            loss_sum += batch_size * loss_batch.item()
            
            # tqdm
            if verbose:
                dataloader_tqdm.set_description(f"eval set: "
                                                f"batch={n_batch}/{n_batches}, "
                                                f"loss_batch={loss_batch.item():.4f}, "
                                                f"acc_batch={acc_batch.item():.4f}")
            else:
                pass
            
    # compute loss test
    loss = loss_sum / n_data
    # accuracy test
    accuracy = correct_predictions_sum / n_data
    
    # loss per instance
    losses = torch.reshape(torch.tensor(np.concatenate(losses)), (n_data,))
    # softmax prob vector per instance
    softmaxes = torch.tensor(np.concatenate(softmaxes))
    # predicitons per instance
    predictions = torch.reshape(torch.tensor(np.concatenate(predictions)), (n_data,))
    
    return loss, accuracy, losses, softmaxes, predictions


def get_lc_params(model_ema, train_eval_dataloader, device, n_epoch, n_epochs, verbose=True):
    """ Getting lc params """
    # don't change model params, eval mode
    # notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode
    model_ema.eval()
    softmaxes = []
    losses = []
    n_data = len(train_eval_dataloader.dataset)
    
    # tqdm
    n_batches = len(train_eval_dataloader)
    if verbose:
        train_eval_dataloader_tqdm = tqdm(enumerate(train_eval_dataloader), total=n_batches,
                                         file=sys.stdout, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        train_eval_dataloader_tqdm.set_description(f"epoch={n_epoch}/{n_epochs}, getting lc params")
    else:
        pass
    
    # no backprop
    # impacts the autograd engine and deactivate it. It will reduce memory usage and speed 
    # up computations but you won’t be able to backprop (which you don’t want in an eval script).
    with torch.no_grad():
        for n_batch, (data_batch, targets_batch_one_hot, idx_batch) in train_eval_dataloader_tqdm:
            # since zero-indexed
            n_batch += 1
            # put tensors onto device
            data_batch, targets_batch_one_hot = data_batch.to(device), targets_batch_one_hot.to(device)
            # get model logits
            logits_batch = model_ema(data_batch)
            # get loss
            loss_batch = -torch.sum(F.log_softmax(logits_batch, dim=1)*targets_batch_one_hot, dim=1)
            # get softmax from logits
            softmax_batch = F.softmax(logits_batch, dim=1)
            #predictions_batch = softmax_batch.argmax(axis=1)
            # If the tensor is on a device other than "cpu", you will need to bring it back to the CPU before you can call the .numpy() method. 
            # append to lists
            losses.append(loss_batch.to("cpu").numpy())
            softmaxes.append(softmax_batch.to("cpu").numpy())
            
            # tqdm
            if verbose:
                train_eval_dataloader_tqdm.set_description(f"epoch={n_epoch}/{n_epochs}, getting lc params, batch={n_batch}/{n_batches}")
            else:
                pass
    
    # loss per instance
    losses = torch.reshape(torch.tensor(np.concatenate(losses)), (n_data,))
    # softmax prob vector per instance
    softmaxes = torch.tensor(np.concatenate(softmaxes))
    
    return losses, softmaxes
