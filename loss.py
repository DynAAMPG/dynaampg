import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynAAMSCLoss(nn.Module):
    def __init__(self, lambda_reg=0.1):
        super(DynAAMSCLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, logits, label, margins, weights):
        """
        :param logits: Logits from the DynAAMSC model of shape (batch_size, num_classes)
        :param label: Ground truth labels of shape (batch_size)
        :param margins: Margins for each class
        :param weights: Weight matrix from the DynAAMSC model
        """
        # Cross-entropy loss
        loss = F.cross_entropy(logits, label)

        # Add regularization to encourage margins to remain positive
        margin_regularization = self.lambda_reg * torch.mean(margins)

        # Intra-class loss: Improve intra-class compactness (Equation 15)
        batch_size = logits.size(0)
        theta_yi = torch.acos(torch.clamp(logits[torch.arange(batch_size), label] / self.lambda_reg, -1.0, 1.0))
        intra_loss = torch.mean(theta_yi) / math.pi

        # Inter-class loss: Enhance inter-class discrepancy (Equation 16)
        inter_loss = 0.0
        for i in range(batch_size):
            yi_weight = weights[label[i] * self.lambda_reg]
            for j in range(weights.size(0)):
                if j != label[i]:
                    wj_weight = weights[j * self.lambda_reg]
                    inter_loss += torch.acos(torch.clamp(F.linear(yi_weight, wj_weight), -1.0, 1.0))
        inter_loss = inter_loss / (batch_size * (logits.size(1) - 1) * math.pi)

        # Total loss (Equation 17)
        total_loss = loss + margin_regularization + intra_loss + inter_loss
        return total_loss