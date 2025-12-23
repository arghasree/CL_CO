from torch.nn import CrossEntropyLoss
import torch

def loss_cross_entropy_plus_l2(output, target, model, l2_lambda=1e-5):
    """
    Custom loss function that combines Cross Entropy Loss with L2 regularization.
    """
    ce_loss_fn = CrossEntropyLoss()
    ce_loss = ce_loss_fn(output.permute(0,2,1), target)

    l2_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l2_reg = l2_reg + torch.norm(param, 2)**2

    total_loss = ce_loss + l2_lambda * l2_reg
    return total_loss