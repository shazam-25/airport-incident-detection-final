import os
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from typing import Dict, List, Any

from src.models.config import MultiTaskModelConfig
from src.models.network import MultiTaskYOLO

class MultiTaskEvaluator:
    """Evaluates multi-tak YOLO model on validation/text datasets.
    Computes per-task mAP@50, mAP@50-95, mAR, and class-wise performance."""
    def __init__(
        self,
        model: MultiTaskYOLO,
        config: MultiTaskModelConfig = MultiTaskModelConfig(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        self.task_names = config.get_head_names()

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> Dict[str, Dict[str, Any]]:
        """Runs evaluation across all batches and returns per-task evaluation metrics."""
        print(f"🔍 Starting Multi-Task Evaluation on [{self.device.upper()}]...")
        
        # Initialize TorchMetrics mAP engine per task
        map_metric_calculators = {
            task_name: MeanAveragePrecision(
                box_format="cxcywh", # YOLO format outputs center_x, center_y, width, height
                iou_type="bbox",
                class_metrics=True
            ).to(self.device)
            for task_name in self.task_names
        }

        task_map = {0: "turnaround", 1: "ppe", 2: "fod"}

        for images, targets, task_ids in tqdm(val_loader, desc="Evaluating Batches"):
            images = images.to(self.device, non_blocking=True)
            task_ids = task_ids.to(self.device, non_blocking=True)

            # Model Forward Pass
            # Outputs dict: {task_name: {"predictions": decoded_preds}}
            predictions = self.model(images, task_ids)

            for batch_idx, task_id in enumerate(task_ids.tolist()):
                task_name = task_map[task_id]
                if task_name not in predictions:
                    continue

                # 1. Process Ground Truth Bounding Boxes for sample
                sample_gt_boxes = targets.get(task_name, torch.zeros((0, 6)))
                # Filter rows belonging to current batch element within task slice
                mask = sample_gt_boxes[:, 0] == batch_idx if len(sample_get_boxes) > 0 else []
                gt_slice = sample_gt_boxes[mask] if len(mask) > 0 else torch.zeros((0,6))

                gt_dict = [
                    {
                        "boxes": gt_slice[:, 2:6].to(self.device) if len(gt_slice) > 0 else torch.zeros((0,4), device=self.device)
                        "labels": gt_slice[:, 1].long().to(self.device) if len(gt_slice) > 0 else torch.zeros((0,), dtype=torch.long, device=self.device)
                    }
                ]

                # 2. Extract Raw Predictions for task head
                raw_preds = predictions[task_name]["predictions"]

                # If raw multi-scale feature list or formatted predictions tensor
                if isinstance(raw_preds, (list, tuple)):
                    pred_boxes = torch.zeros((0, 4), device=self.device)
                    pred_scores = torch.zeros((0,), device=self.device)
                    pred_labels = torch.zeros((0,), dtype=torch.long, device=self.device)
                else: 
                    # Assuming decoded box format [boxes, scores, labels]
                    pred_boxes = raw_preds[batch_idx, :, :4] if raw_preds.ndim == 3 else torch.zeros((0, 4), device=self.device)
                    pred_score = raw_preds[batch_idx, :, 4] if raw_preds.ndim == 3 else torch.zeros((0,), device=self.device)
                    pred_labels = raw_pred[batch-idx, :, 5].long() if raw_preds.ndim == 3 else torch.zeros((0,), device=self.device, dtype=torch.long)

                    # Apply confidence threshold filtering
                    conf_mask = pred_scores >= conf_thresh
                    pred_boxes = pred_boxes[conf_mask]
                    pred_scores = pred_scores[conf_mask]
                    pred_labels = pred_labels[conf_mask]

                pred_dict = [
                    {
                        "boxes": pred_boxes,
                        "scores": pred_scores,
                        "labels": pred_labels
                    }
                ]

                # Update final metrics per task
                results = {}
                print("\n" + "="*55)
                print("📊 MULTI-TASK EVALUATION METRICS REPORT")
                print("="*55)

                for task_name, metric_calc in map_metric_calculators.items():
                    metrics = metric_calc.compute()
                    results[task_name] = {
                        "mAP_50": metrics["map_50"].item(),
                        "mAP_50_95": metrics["map"].item(),
                        "mar_100": metrics["mar_100"].item()
                    }

                    rint(f"\n🔹 STREAM HEAD: [{task_name.upper()}]")
            print(f"  • mAP @ 0.50     : {results[task_name]['mAP_50']:.4f}")
            print(f"  • mAP @ 0.50:0.95: {results[task_name]['mAP_50_95']:.4f}")
            print(f"  • mAR @ 100      : {results[task_name]['mar_100']:.4f}")

        print("\n" + "="*55 + "\n")
        return results


def run_evaluation(checkpoint_path: str, test_data_config: dict, batch_size: int = 16):
    """Utility wrapper function to load checkpoint and trigger validation evaluation."""
    from src.data.dataloader import create_multitask_dataloader

    config = MultiTaskModelConfig()
    model = MultiTaskYOLO(config)

    # Load trained checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded checkpoint weights from: {checkpoint_path}")
    else:
        print(f"⚠️ Warning: Checkpoint path '{checkpoint_path}' not found. Evaluating on random/pretrained weights.")

    test_loader = create_multitask_dataloader(
        data_dirs=test_data_config,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    evaluator = MultiTaskEvaluator(model=model, config=config)
    return evaluator.evaluate(test_loader)