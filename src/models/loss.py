from types import SimpleNamespace
import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss
from src.models.config import MultiTaskModelConfig

class HeadLossAdapter(nn.Module):
    def __init__(self, task_head, default_args):
        super().__init__()
        self.args = default_args
        self.model = nn.ModuleList([task_head.detect])


class TaskLossWrapper(nn.Module):
    """
    Wraps Ultralytics v8DetectionLoss, adapts target tensors, and ensures internal
    loss parameters and assigners (proj, stride, assigner) are transferred to GPU device.
    """
    def __init__(self, task_head, default_args):
        super().__init__()
        adapter = HeadLossAdapter(task_head, default_args)
        self.loss_fn = v8DetectionLoss(adapter)

    def _get_device(self, predictions):
        if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
            item = predictions[0]
            if isinstance(item, (list, tuple)) and len(item) > 0:
                return item[0].device
            elif isinstance(item, torch.Tensor):
                return item.device
        elif isinstance(predictions, dict):
            for v in predictions.values():
                if isinstance(v, torch.Tensor):
                    return v.device
                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    return v[0].device
        elif isinstance(predictions, torch.Tensor):
            return predictions.device
        
        return getattr(self.loss_fn, "device", torch.device("cpu"))

    def forward(self, predictions, target_tensor):
        device = self._get_device(predictions)

        # Ensure internal v8DetectionLoss device attribute matches
        self.loss_fn.device = device

        # Transfer internal loss attributes and tensors to GPU
        if hasattr(self.loss_fn, "proj") and isinstance(self.loss_fn.proj, torch.Tensor):
            if self.loss_fn.proj.device != device:
                self.loss_fn.proj = self.loss_fn.proj.to(device)

        if hasattr(self.loss_fn, "stride") and isinstance(self.loss_fn.stride, torch.Tensor):
            if self.loss_fn.stride.device != device:
                self.loss_fn.stride = self.loss_fn.stride.to(device)

        # Transfer internal TaskAlignedAssigner to GPU if present
        if hasattr(self.loss_fn, "assigner") and self.loss_fn.assigner is not None:
            if hasattr(self.loss_fn.assigner, "device") and self.loss_fn.assigner.device != device:
                self.loss_fn.assigner.device = device


        # Build target dictionary required by v8DetectionLoss
        if target_tensor is None or (isinstance(target_tensor, torch.Tensor) and target_tensor.numel() == 0):
            target_dict = {
                "batch_idx": torch.zeros((0, 1), device=device),
                "cls": torch.zeros((0, 1), device=device),
                "bboxes": torch.zeros((0, 4), device=device),
            }
        else:
            target_dict = {
                "batch_idx": target_tensor[:, 0:1].to(device),
                "cls": target_tensor[:, 1:2].to(device),
                "bboxes": target_tensor[:, 2:6].to(device),
            }

        return self.loss_fn(predictions, target_dict)


class MultiTaskUncertaintyLoss(nn.Module):
    def __init__(self, model: nn.Module, config: MultiTaskModelConfig = MultiTaskModelConfig()):
        super().__init__()
        self.config = config
        self.task_names = config.get_head_names()

        default_args = SimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
            reg_max=16
        )

        self.task_losses = nn.ModuleDict({
            name: TaskLossWrapper(head_module, default_args)
            for name, head_module in model.heads.items()
        })

        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, requires_grad=True))
            for name in self.task_names
        })

    def forward(self, predictions: dict, targets: dict):
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        loss_items = {}

        for task_name in predictions.keys():
            if task_name not in targets or targets[task_name] is None:
                continue

            task_pred = predictions[task_name]["predictions"]
            task_target = targets[task_name]

            # Compute raw task loss
            raw_loss, loss_components = self.task_losses[task_name](task_pred, task_target)

            s = self.log_vars[task_name]
            precision = torch.exp(-s)

            weighted_loss = precision * raw_loss.sum() + 0.5 * s
            total_loss = total_loss + weighted_loss

            loss_items[f"loss_{task_name}_raw"] = raw_loss.sum().detach()
            loss_items[f"loss_{task_name}_weighted"] = weighted_loss.detach()
            loss_items[f"sigma_{task_name}"] = torch.exp(0.5 * s).detach()

        return total_loss, loss_items

# from types import SimpleNamespace
# import torch
# import torch.nn as nn
# from ultralytics.utils.loss import v8DetectionLoss
# from src.models.config import MultiTaskModelConfig

# class HeadLossAdapter(nn.Module):
#     """
#     Adapter module that mimics Ultralytics model structure (model.model[-1])
#     required by v8DetectionLoss.
#     """
#     def __init__(self, task_head, default_args):
#         super().__init__()
#         self.args = default_args
#         # Ultralytics v8DetectionLoss expects model.model[-1] to be the Detect module
#         self.model = nn.ModuleList([task_head.detect])

# class TaskLossWrapper(nn.Module):
#     """
#     Wraps Ultralytics v8DetectionLoss inside a standard PyTorch nn.Module 
#     so it can be safely stored in an nn.ModuleDict.
#     """
#     def __init__(self, task_head, default_args):
#         super().__init__()
#         adapter = HeadLossAdapter(task_head, default_args)
#         self.loss_fn = v8DetectionLoss(adapter)

#     def _get_device(self, predictions):
#         # Extract device from predictions (list/tuple of tensors or tensor directly)
#         if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
#             item = predictions[0]
#             if isinstance(item, (list, tuple)) and len(item) > 0:
#                 return item[0].device
#             elif isinstance(item, torch.Tensor):
#                 return item.device
#         elif isinstance(predictions, dict):
#             for v in predictions.values():
#                 if isinstance(v, torch.Tensor):
#                     return v.device
#                 elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
#                     return v[0].device
#         elif isinstance(predictions, torch.Tensor):
#             return predictions.device
        
#         # Fallback to device attribute on loss function or CPU
#         return getattr(self.loss_fn, "device", torch.device("cpu"))

#     def forward(self, predictions, target_tensor):
#         device = self._get_device(predictions)

#         # CRITICAL FIX: Ensure internal loss parameters (like self.proj) match execution device
#         if hasattr(self.loss_fn, "proj") and isinstance(self.loss_fn.proj, torch.Tensor):
#             if self.loss_fn.proj.device != device:
#                 self.loss_fn.proj = self.loss_fn.proj.to(device)

#         # Build target dictionary required by v8DetectionLoss
#         if target_tensor is None or (isinstance(target_tensor, torch.Tensor) and target_tensor.numel() == 0):
#             target_dict = {
#                 "batch_idx": torch.zeros((0, 1), device=device),
#                 "cls": torch.zeros((0, 1), device=device),
#                 "bboxes": torch.zeros((0, 4), device=device),
#             }
#         else:
#             target_dict = {
#                 "batch_idx": target_tensor[:, 0:1].to(device),
#                 "cls": target_tensor[:, 1:2].to(device),
#                 "bboxes": target_tensor[:, 2:6].to(device),
#             }

#         return self.loss_fn(predictions, target_dict)

# class MultiTaskUncertaintyLoss(nn.Module):
#     """
#     Multi-Task Loss Wrapper with Learnable Homoscedastic Uncertainty Weighting
#     (Kendall et al., CVPR 2018). Automatically balances loss gradients across
#     Turnaround, PPE, and FOD tasks during backpropagation.
#     """
#     def __init__(self, model: nn.Module, config: MultiTaskModelConfig = MultiTaskModelConfig()):
#         super().__init__()
#         self.config = config
#         self.task_names = config.get_head_names()
        
#         # Standard YOLOv8 loss hyperparameters
#         default_args = SimpleNamespace(
#             box=7.5,    # Box loss gain
#             cls=0.5,    # Class loss gain
#             dfl=1.5,    # Distribution Focal Loss gain
#             reg_max=16
#         )

#         # 1. Initialize YOLO detection loss modules inside PyTorch ModuleDict
#         self.task_losses = nn.ModuleDict({
#             name: TaskLossWrapper(head_module, default_args) for name, head_module in model.heads.items()
#         })

#         # 2. Learnable task log variances s_i = log(sigma_i^2)
#         self.log_vars = nn.ParameterDict({
#             name: nn.Parameter(torch.zeros(1, requires_grad=True))
#             for name in self.task_names
#         })

#     def forward(self, predictions: dict, targets: dict):
#         """
#         Args:
#             predictions (dict): Output from MultiTaskYOLO forward pass.
#             targets (dict): Target ground truth batch dict structured by task.
#         """
#         total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
#         loss_items = {}

#         for task_name in predictions.keys():
#             if task_name not in targets or targets[task_name] is None:
#                 continue
                
#             task_pred = predictions[task_name]["predictions"]
#             task_target = targets[task_name]
            
#             # Compute raw task detection loss via wrapped loss module
#             raw_loss, loss_components = self.task_losses[task_name](task_pred, task_target)
            
#             # Extract log variance parameter s = log(sigma^2)
#             s = self.log_vars[task_name]
#             precision = torch.exp(-s)
            
#             # Weighted loss calculation with uncertainty penalty
#             weighted_loss = precision * raw_loss.sum() + 0.5 * s
            
#             total_loss = total_loss + weighted_loss
            
#             # Record metrics for tracking
#             loss_items[f"loss_{task_name}_raw"] = raw_loss.sum().detach()
#             loss_items[f"loss_{task_name}_weighted"] = weighted_loss.detach()
#             loss_items[f"sigma_{task_name}"] = torch.exp(0.5 * s).detach()

#         return total_loss, loss_items