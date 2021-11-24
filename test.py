from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np


def test(model, device, dataloader):
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
    dataloader_tqdm = tqdm(enumerate(dataloader), total=n_batches)
    dataloader_tqdm.set_description(f"\teval set")
    
    with torch.no_grad():
        for n_batch, (data_batch, targets_batch_one_hot) in dataloader_tqdm:
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
            dataloader_tqdm.set_description(f"\teval set: "
                                            f"batch={n_batch}/{n_batches}, "
                                            f"loss_batch={loss_batch.item():.4f}, "
                                            f"acc_batch={acc_batch.item():.4f}")
            
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
