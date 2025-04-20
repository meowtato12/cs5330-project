import os
import torch
import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt  # New: for PR curve plotting

# Function to compute Intersection over Union (IoU) between two bounding boxes
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_t, y1_t, x2_t, y2_t = box2
    xi1 = max(x1, x1_t)
    yi1 = max(y1, y1_t)
    xi2 = min(x2, x2_t)
    yi2 = min(y2, y2_t)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_t - x1_t) * (y2_t - y1_t)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Function to evaluate a trained model and visualize predictions
def evaluate_model(model, data_loader, device, output_dir, score_threshold=0.3):
    model.eval()
    all_scores = []
    all_labels = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            images, targets, orig_images = batch
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for img, output, target, orig_img in zip(images, outputs, targets, orig_images):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                true_boxes = target['boxes'].cpu().numpy()
                filename = target['filename']

                valid_mask = pred_scores >= score_threshold
                pred_boxes = pred_boxes[valid_mask]
                pred_scores = pred_scores[valid_mask]

                img_with_boxes = orig_img.copy()
                for box in true_boxes:
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                for box, score in zip(pred_boxes, pred_scores):
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"{score:.2f}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                output_path = os.path.join(output_dir, f"annotated_{filename.replace('.tiff', '.jpg')}")
                cv2.imwrite(output_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))

                for i, pred_box in enumerate(pred_boxes):
                    iou_max = 0
                    for true_box in true_boxes:
                        iou = compute_iou(pred_box, true_box)
                        iou_max = max(iou_max, iou)
                    if iou_max > 0.5:
                        all_labels.append(1)
                    else:
                        all_labels.append(0)
                    all_scores.append(pred_scores[i])

    if not all_labels or not all_scores:
        return 0.0

    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    pr_auc = auc(recall, precision)

    # New: Save PR Curve plot
    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)

    output_img_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(output_img_path)
    plt.close()

    return pr_auc

