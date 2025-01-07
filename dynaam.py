import torch
import torch.nn as nn
import torch.nn.functional as F

class DynAAM(nn.Module):
    def __init__(self, in_features, out_features, num_sub_centers=3, scale=64.0, base_margin=0.3, lambda_reg=0.1):
        """
        :param in_features: Number of input features (e.g., feature embedding size)
        :param out_features: Number of classes
        :param num_sub_centers: Number of sub-centers per class
        :param scale: Scale factor, often denoted as 's'
        :param base_margin: Initial margin value for each class, often denoted as 'm'
        :param lambda_reg: Regularization parameter to control margin smoothness
        """
        super(DynAAM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_sub_centers = num_sub_centers
        self.scale = scale
        self.base_margin = base_margin
        self.lambda_reg = lambda_reg

        # Learnable margins for each class
        self.margins = nn.Parameter(torch.ones(out_features) * base_margin)
        # Learnable weight matrix with sub-centers
        self.weights = nn.Parameter(torch.randn(out_features * num_sub_centers, in_features))

        # Weight initialization
        nn.init.xavier_uniform_(self.weights)

    def forward(self, input, label):
        """
        :param input: Input features of shape (batch_size, in_features)
        :param label: Ground truth labels of shape (batch_size)
        """
        # Normalize input and weight matrix
        input = F.normalize(input, p=2, dim=1)
        weight = F.normalize(self.weights, p=2, dim=1)

        batch_size = input.size(0)
        num_classes = self.out_features
        num_sub_centers = self.num_sub_centers

        # Compute cosine similarities between input and weight (multiple sub-centers per class)
        cosine_similarities = F.linear(input, weight)  # Shape: (batch_size, num_classes * num_sub_centers)
        cosine_similarities = cosine_similarities.view(batch_size, num_classes, num_sub_centers)
        
        # Use max pooling to select the most appropriate sub-center per class
        max_cosine, _ = torch.max(cosine_similarities, dim=2)  # Shape: (batch_size, num_classes)

        # Get margins corresponding to the ground truth labels
        batch_margins = self.margins[label]  # Shape: (batch_size)

        # Calculate cosine values with dynamic margin
        target_cosine = max_cosine[torch.arange(batch_size), label]  # Shape: (batch_size)
        cosine_with_margin = target_cosine - batch_margins

        # Update the cosine similarities with the margin for the target class
        max_cosine[torch.arange(batch_size), label] = cosine_with_margin

        # Equation (6) implementation: Calculate the average angular shift of sub-centers
        avg_angular_shift = torch.zeros(batch_size)
        for i in range(batch_size):
            class_label = label[i]
            main_center_angle = cosine_similarities[i, class_label, 0]  # Dominant center (assuming c1 is index 0)
            sub_center_angles = cosine_similarities[i, class_label, 1:]  # All other sub-centers
            avg_shift = torch.sum(torch.abs(main_center_angle - sub_center_angles)) / (num_sub_centers - 1)
            avg_angular_shift[i] = avg_shift

        # Weight the margin with influence of each sub-center dispersion
        weighted_margin = batch_margins * (1 + avg_angular_shift)
        cosine_with_margin = target_cosine - weighted_margin

        # Update the cosine similarities with the new weighted margin for the target class
        max_cosine[torch.arange(batch_size), label] = cosine_with_margin

        # Scale the result
        logits = self.scale * max_cosine

        return logits