import os
from pathlib import Path
import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm

from src.models.config import MultiTaskModelConfig
from src.models.network import MultiTaskYOLO
from src.models.loss import MultiTaskUncertaintyLoss

# Automatically identify project root directory (two levels up from src/models/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class MultiTaskTrainer:
    """Trainer engine for Multi-Task YOLO architecture with AMP, Gradient Clipping,
    TensorBoard / WandB experiment tracking."""
    def __init__(
        self,
        model: MultiTaskYOLO,
        loss_fn: MultiTaskUncertaintyLoss,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        config: MultiTaskModelConfig = MultiTaskModelConfig(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_wandb: bool = False,
        exp_name: str = "multitask_yolo_run",
        checkpoint_dir: str = "checkpoints",
        grad_clip_norm: float = 10.0
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = PROJECT_ROOT / "model" / checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Automatic Mixed Precision Scaler
        self.scaler = GradScaler(device=self.device)

        # TensorBoard Writer
        self.tb_writer = SummaryWriter(log_dir=os.path.join(PROJECT_ROOT, "model", "runs", exp_name))

        # Weights & Biases Setup
        if self.use_wandb:
            wandb.init(project="airport-incident-detection-final", name=exp_name)

        self.best_val_loss = float("inf")
    
    def train_epoch(self, epoch: int):
        self.model.train()
        self.loss_fn.train()

        running_total_loss = 0.0
        loss_tracker = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch_idx, (images, targets, task_ids) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            task_ids = task_ids.to(self.device, non_blocking=True)

            # Transfer target tensor to device
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
            
            self.optimizer.zero_grad()

            # Forward pass with AMP
            with autocast(device_type=self.device):
                predictions = self.model(images, task_ids)
                loss, loss_items = self.loss_fn(predictions, targets)

            # Backward pass with Scaler
            self.scaler.scale(loss).backward()

            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            # Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate metrics
            running_total_loss += loss.item()
            for key, val in loss_items.items():
                loss_tracker[key] = loss_tracker.get(key, 0.0) + val.item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        num_batches = len(self.train_loader)
        avg_train_loss = running_total_loss / num_batches
        avg_loss_items = {k: v / num_batches for k, v in loss_tracker.items()}

        return avg_train_loss, avg_loss_items

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        self.loss_fn.eval()

        running_val_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

        for images, targets, task_ids in pbar:
            images = images.to(self.device, non_blocking=True)
            task_ids = task_ids.to(self.device, non_blocking=True)

            if isinstance(targets, dict):
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

            with autocast(device_type=self.device):
                predictions = self.model(images, task_ids)
                loss, _ = self.loss_fn(predictions, targets)

            running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(self.val_loader)
        return avg_val_loss

    def fit(self, num_epochs: int):
        print(f"🚀 Starting Multi-Task Model Training on device [{self.device.upper()}] for {num_epochs} Epcohs...\n")

        for epoch in range(1, num_epochs + 1):
            train_loss, loss_items = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            if self.scheduler:
                self.scheduler.step()

            # Log to TensorBoard
            self.tb_writer.add_scalar("Loss/Train_Total", train_loss, epoch)
            self.tb_writer.add_scalar("Loss/Val_Total", val_loss, epoch)

            for item_name, item_val in loss_items.items():
                self.tb_writer.add_scalar(f"Metrics/{item_name}", item_val, epoch)

            # Log to WandB
            if self.use_wandb:
                wandb_log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": self.optimizer.param_groups[0]["lr"]
                }
                wandb_log_dict.update(loss_items)
                wandb.log(wandb_log_dict)

            print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_multitask_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss_state_dict": self.loss_fn.state_dict(),
                    "val_loss": val_loss
                }, checkpoint_path)
                print(f" 💾 Saved Best Model Checkpoint (Val LOss: {val_loss:.4f}) -> {checkpoint_path}")

            self.tb_writer.close()
            if self.use_wandb:
                wandb.finish()

            print("\n✅ Multi-Task Training Pipeline Completed Successfully!")



