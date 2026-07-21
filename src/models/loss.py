from types import SimpleNamespace
import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss
from src.models.config import MultiTaskModelConfig

class HeadLossAdapter(nn.Module):
    """
    Adapter module that mimics Ultralytics model structure (model.model[-1])
    required by v8DetectionLoss.
    """
    def __init__(self, task_head, default_args):
        super().__init__()
        self.args = default_args
        # Ultralytics v8DetectionLoss expects model.model[-1] to be the Detect module
        self.model = nn.ModuleList([task_head.detect])

class TaskLossWrapper(nn.Module):
    """
    Wraps Ultralytics v8DetectionLoss inside a standard PyTorch nn.Module 
    so it can be safely stored in an nn.ModuleDict.
    """
    def __init__(self, task_head, default_args):
        super().__init__()
        adapter = HeadLossAdapter(task_head, default_args)
        self.loss_fn = v8DetectionLoss(adapter)

    def forward(self, predictions, targets):
        return self.loss_fn(predictions, targets)

class MultiTaskUncertaintyLoss(nn.Module):
    """
    Multi-Task Loss Wrapper with Learnable Homoscedastic Uncertainty Weighting
    (Kendall et al., CVPR 2018). Automatically balances loss gradients across
    Turnaround, PPE, and FOD tasks during backpropagation.
    """
    def __init__(self, model: nn.Module, config: MultiTaskModelConfig = MultiTaskModelConfig()):
        super().__init__()
        self.config = config
        self.task_names = config.get_head_names()
        
        # Standard YOLOv8 loss hyperparameters
        default_args = SimpleNamespace(
            box=7.5,    # Box loss gain
            cls=0.5,    # Class loss gain
            dfl=1.5,    # Distribution Focal Loss gain
            reg_max=16
        )

        # 1. Initialize YOLO detection loss modules inside PyTorch ModuleDict
        self.task_losses = nn.ModuleDict()
        for name, head_module in model.heads.items():
            self.task_losses[name] = TaskLossWrapper(head_module, default_args)

        # 2. Learnable task log variances s_i = log(sigma_i^2)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, requires_grad=True))
            for name in self.task_names
        })

    def forward(self, predictions: dict, targets: dict):
        """
        Args:
            predictions (dict): Output from MultiTaskYOLO forward pass.
            targets (dict): Target ground truth batch dict structured by task.
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        loss_items = {}

        for task_name in predictions.keys():
            if task_name not in targets or targets[task_name] is None:
                continue
                
            task_pred = predictions[task_name]["predictions"]
            task_target = targets[task_name]
            
            # Compute raw task detection loss via wrapped loss module
            raw_loss, loss_components = self.task_losses[task_name](task_pred, task_target)
            
            # Extract log variance parameter s = log(sigma^2)
            s = self.log_vars[task_name]
            precision = torch.exp(-s)
            
            # Weighted loss calculation with uncertainty penalty
            weighted_loss = precision * raw_loss.sum() + 0.5 * s
            
            total_loss = total_loss + weighted_loss
            
            # Record metrics for tracking
            loss_items[f"loss_{task_name}_raw"] = raw_loss.sum().detach()
            loss_items[f"loss_{task_name}_weighted"] = weighted_loss.detach()
            loss_items[f"sigma_{task_name}"] = torch.exp(0.5 * s).detach()

        return total_loss, loss_items