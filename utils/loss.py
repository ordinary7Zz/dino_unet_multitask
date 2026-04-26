import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

def structure_loss(pred, mask):
    # ----------- weighted BCE -------------
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # ----------- weighted IoU -------------
    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def benign_malignant_loss(predict, target, pos_weight=None):
    """
    良恶性二分类损失
    predict: (B, 1) - logits
    target: (B,) 或 (B,1) - 0/1
    pos_weight: 对恶性样本(M=1)加权，可以是标量或tensor
                例如: 2.0 或 tensor([2.0])
                pos_weight > 1 增加对恶性样本(正类)的重视,提高Recall
    """
    if target.dim() == 2:
        target = target.squeeze(1)

    # 如果pos_weight不是tensor,转换为tensor
    if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor([pos_weight], device=predict.device, dtype=torch.float32)
    elif pos_weight is not None:
        # 确保pos_weight在正确的设备上
        pos_weight = pos_weight.to(predict.device)

    loss = F.binary_cross_entropy_with_logits(
        predict.squeeze(1),
        target.float(),
        pos_weight=pos_weight
    )
    return loss


def tirads_loss(predict, target, class_weights=None, label_smoothing=0.0):
    """
    TI-RADS 5分类损失
    predict: (B, 5) - logits
    target: (B,) - int64 class index
    class_weights: tensor([w1, w2, ..., w5])
    label_smoothing: optional smoothing
    """
    loss = F.cross_entropy(
        predict,
        target,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction='mean'
    )
    return loss

def benign_malignant_loss_gla(predict, target, p_pos, p_neg, tau=1.0):
    """
    增强版 GLA 良恶性二分类损失（支持Recall提升）
    
    predict: (B, 1) - logits
    target: (B,) 或 (B,1) - 0/1 (0: benign, 1: malignant)
    p_pos: 恶性样本比例 p(y=1)
    p_neg: 良性样本比例 p(y=0)
    tau: GLA调整系数 τ（默认1）
    """

    if target.dim() == 2:
        target = target.squeeze(1)

    # ====== Logit Adjustment (GLA) ======
    # 添加数值稳定性保护，确保概率值不为0
    epsilon = 1e-8
    p_pos = max(p_pos, epsilon)
    p_neg = max(p_neg, epsilon)
    
    logit_adjust = target * torch.log(torch.tensor(p_pos, device=predict.device)) + \
                   (1 - target) * torch.log(torch.tensor(p_neg, device=predict.device))

    adjusted_logits = predict.squeeze(1) + tau * logit_adjust
    
    # 添加数值稳定性保护，防止logits过大或过小
    adjusted_logits = torch.clamp(adjusted_logits, min=-100, max=100)

    loss = F.binary_cross_entropy_with_logits(
        adjusted_logits,
        target.float()
    )
    return loss


def tirads_loss_gla(predict, target, p_class, tau=1.0, label_smoothing=0.0):
    """
    基于 GLA 的 TI-RADS 5类分类损失（Multiclass Logit-Adjusted Loss）

    predict: (B,5) logits
    target: (B,)
    p_class: tensor([p1,p2,p3,p4,p5]) 类别频率
    tau: GLA 调整系数
    """

    # ====== Logit Adjustment ======
    # 添加数值稳定性保护，确保概率值不为0
    epsilon = 1e-8
    p_class = torch.clamp(p_class, min=epsilon, max=1.0 - epsilon)
    
    # shape: (5,) → broadcast 到 (B,5)
    log_p = torch.log(p_class.to(predict.device))
    adjusted_logits = predict + tau * log_p
    
    # 添加数值稳定性保护，防止logits过大或过小
    adjusted_logits = torch.clamp(adjusted_logits, min=-100, max=100)

    # ====== Cross Entropy ======
    loss = F.cross_entropy(
        adjusted_logits,
        target,
        label_smoothing=label_smoothing,
        reduction='mean'
    )
    return loss
