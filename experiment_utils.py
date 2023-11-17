import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader


import text_cnn as cnn


def get_roc_auc(labels:torch.Tensor, pred_labels:torch.Tensor):
    return roc_auc_score(labels.detach().cpu(),
                         pred_labels.softmax(1)[:, 1].detach().cpu())


def get_acc(labels:torch.Tensor, pred_labels:torch.Tensor):
    return (
        (pred_labels.argmax(1) == labels)
        .sum()
        .item()
        /pred_labels.shape[0]
    )


def train(dataloader:DataLoader, model:cnn.TextCnn,
          optimizer:torch.optim.Adam, criterion:nn.CrossEntropyLoss, epoch:int,
          log_interval:int=1000) -> torch.Tensor:
    model.train()

    total_loss = 0.
    for idx, (nums, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        pred_labels = model.forward(nums)
        loss = criterion.forward(pred_labels, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_loss = total_loss + loss.detach().cpu()
        if(len(labels.unique()) > 1):
            try:
                auc = get_roc_auc(labels, pred_labels)
            except:
                import pdb
                pdb.set_trace()
            acc = get_acc(labels, pred_labels)
        if idx % log_interval == 0 and idx > 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| auc {:8.3f}'
                  '| acc {:8.3f}'.format(epoch, idx, len(dataloader),
                                         auc, acc)
                 )
    
    return total_loss


def evaluate(dataloader:DataLoader, model:torch.nn.Module, 
             criterion:nn.CrossEntropyLoss=None) -> tuple:
    preds, labels = predict(dataloader, model)

    if criterion is not None:
        with torch.no_grad():
            loss = criterion.forward(preds, labels.long()).detach().cpu().numpy()
    else:
        loss = None

    return (
        get_roc_auc(labels, preds), 
        get_acc(labels, preds),
        loss
    )


def predict(dataloader:DataLoader, model:torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    n_samples = len(dataloader.dataset)
    labels_all = torch.zeros(n_samples)
    preds_all = torch.zeros((n_samples, 2))

    def to_cpu(t:torch.Tensor) -> np.ndarray:
        return t.detach().cpu()

    with torch.no_grad():
        current_start = 0
        for nums, labels in dataloader:
            pred_labels:torch.Tensor = model(nums)
            current_stop = current_start + pred_labels.size(0)
            preds_all[current_start:current_stop, :] = to_cpu(pred_labels)
            labels_all[current_start:current_stop] = to_cpu(labels)
            current_start = current_stop

    return preds_all, labels_all

