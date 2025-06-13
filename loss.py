import torch
import torch.nn.functional as F
import torch
import torch.nn as nn


def masked_cross_entropy_loss(logits, labels, mask, reduction='mean'):
    logits_flat = logits.view(-1, logits.size(-1))  # [batch_size * seq_length, num_classes]

    labels_flat = labels.view(-1)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(logits_flat, labels_flat)

    mask_flat = mask.view(-1).float()
    mask_sum = mask_flat.sum().float()
    if mask_sum == 0:
        return torch.tensor(0.0, device=loss.device)
    loss = loss * mask_flat
    if reduction == 'mean':
        return loss.sum() / mask_flat.sum()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss



def sce_loss(x, y, alpha=1,reduction='mean'):

    y=y.float()
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss




def molecular_denoising_loss(pred, target, lambda_l2=1.0, lambda_cos=0.1):
    l2_loss = F.mse_loss(pred, target, reduction='mean')
    cos_loss = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    total_loss = lambda_l2 * l2_loss + lambda_cos * cos_loss
    return total_loss