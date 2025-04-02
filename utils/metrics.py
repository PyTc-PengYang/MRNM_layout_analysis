import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def evaluate_region_detection(gt_boxes, pred_boxes, gt_labels, pred_labels, iou_threshold=0.5):
    # 匹配预测框和真实框
    matched_indices = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        max_iou = -1
        match_idx = -1

        for pred_idx, pred_box in enumerate(pred_boxes):
            if pred_idx in [idx for _, idx in matched_indices]:
                continue  # 已匹配的预测框跳过

            iou = calculate_iou(gt_box, pred_box)
            if iou > max_iou and iou >= iou_threshold:
                max_iou = iou
                match_idx = pred_idx

        if match_idx != -1:
            matched_indices.append((gt_idx, match_idx))

    # 计算指标
    true_positives = len(matched_indices)
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 计算分类准确率
    correct_classifications = 0
    for gt_idx, pred_idx in matched_indices:
        if gt_labels[gt_idx] == pred_labels[pred_idx]:
            correct_classifications += 1

    classification_accuracy = correct_classifications / true_positives if true_positives > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_accuracy': classification_accuracy
    }


def evaluate_reading_order(gt_reading_order, pred_reading_order):
    # 将阅读顺序转换为边列表
    gt_edges = set([(src, dst) for src, dst in gt_reading_order])
    pred_edges = set([(src, dst) for src, dst in pred_reading_order])

    # 计算准确率
    correct_edges = gt_edges.intersection(pred_edges)
    accuracy = len(correct_edges) / len(gt_edges) if len(gt_edges) > 0 else 0

    return accuracy