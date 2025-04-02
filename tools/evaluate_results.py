import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model results')
    parser.add_argument('--results_file', type=str, required=True, help='path to results JSON file')
    parser.add_argument('--ground_truth_dir', type=str, required=True,
                        help='directory containing ground truth annotations')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='output directory for evaluation results')
    return parser.parse_args()


def calculate_iou(box1, box2):
    # 框的格式为 [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算每个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union = area1 + area2 - intersection

    # 返回IoU
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
        'classification_accuracy': classification_accuracy,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def evaluate_reading_order(gt_reading_order, pred_reading_order):
    # 将阅读顺序转换为边列表
    gt_edges = set([(src, dst) for src, dst in gt_reading_order])
    pred_edges = set([(src, dst) for src, dst in pred_reading_order])

    # 计算准确率
    correct_edges = gt_edges.intersection(pred_edges)
    accuracy = len(correct_edges) / len(gt_edges) if len(gt_edges) > 0 else 0

    # 计算BLEU-2分数 (简化版)
    bleu2 = 0.0
    if len(gt_reading_order) >= 2 and len(pred_reading_order) >= 2:
        # 提取序列
        gt_seq = [src for src, _ in gt_reading_order] + [gt_reading_order[-1][1]]
        pred_seq = [src for src, _ in pred_reading_order] + [pred_reading_order[-1][1]]

        # 计算2-gram精度
        gt_bigrams = set(zip(gt_seq[:-1], gt_seq[1:]))
        pred_bigrams = list(zip(pred_seq[:-1], pred_seq[1:]))

        # 计算匹配的2-gram数量
        matches = sum(1 for bg in pred_bigrams if bg in gt_bigrams)
        precision = matches / len(pred_bigrams) if pred_bigrams else 0

        # 简化的惩罚因子
        brevity_penalty = 1.0 if len(pred_seq) >= len(gt_seq) else np.exp(1 - len(gt_seq) / len(pred_seq))

        bleu2 = brevity_penalty * precision

    return {
        'accuracy': accuracy,
        'bleu2': bleu2
    }


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载预测结果
    with open(args.results_file, 'r') as f:
        results = json.load(f)

    # 评估每个样本
    region_metrics = []
    reading_order_metrics = []

    for result in tqdm(results, desc="Evaluating"):
        image_id = result['image_id']

        # 加载真实标注
        gt_path = os.path.join(args.ground_truth_dir, f"{os.path.splitext(image_id)[0]}.json")
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found for {image_id}")
            continue

        with open(gt_path, 'r') as f:
            gt = json.load(f)

        # 提取真实边界框和标签
        gt_boxes = [region['bbox'] for region in gt['regions']]
        gt_labels = [get_label_id(region['type']) for region in gt['regions']]

        # 提取预测边界框和标签
        pred_boxes = result['boxes']
        pred_labels = result['labels']

        # 评估区域检测
        region_metric = evaluate_region_detection(
            gt_boxes, pred_boxes, gt_labels, pred_labels
        )
        region_metrics.append(region_metric)

        # 评估阅读顺序（如果存在）
        if 'reading_order' in gt and 'reading_order' in result:
            reading_order_metric = evaluate_reading_order(
                gt['reading_order'], result['reading_order']
            )
            reading_order_metrics.append(reading_order_metric)

    # 计算平均指标
    avg_region_metrics = {}
    for metric in region_metrics[0].keys():
        avg_region_metrics[metric] = sum(m[metric] for m in region_metrics) / len(region_metrics)

    avg_reading_order_metrics = {}
    if reading_order_metrics:
        for metric in reading_order_metrics[0].keys():
            avg_reading_order_metrics[metric] = sum(m[metric] for m in reading_order_metrics) / len(
                reading_order_metrics)

    # 保存评估结果
    evaluation_results = {
        'region_detection': avg_region_metrics,
        'reading_order': avg_reading_order_metrics if reading_order_metrics else {}
    }

    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    # 绘制评估图表
    plot_metrics(region_metrics, reading_order_metrics, args.output_dir)

    print("Evaluation completed!")
    print(
        f"Region Detection - Precision: {avg_region_metrics['precision']:.4f}, Recall: {avg_region_metrics['recall']:.4f}, "
        f"F1: {avg_region_metrics['f1']:.4f}, Acc: {avg_region_metrics['classification_accuracy']:.4f}")

    if reading_order_metrics:
        print(
            f"Reading Order - Accuracy: {avg_reading_order_metrics['accuracy']:.4f}, BLEU-2: {avg_reading_order_metrics['bleu2']:.4f}")


def get_label_id(label_type):
    label_map = {
        'Page': 0,
        'Page Number': 1,
        'Main Text': 2,
        'Note': 3,
        'Data': 4,
        'Table': 5,
        'Title': 6
    }
    return label_map.get(label_type, 0)


def plot_metrics(region_metrics, reading_order_metrics, output_dir, gt_labels=None, pred_labels=None):
    # 绘制区域检测指标
    plt.figure(figsize=(12, 6))

    # 提取指标值
    precisions = [m['precision'] for m in region_metrics]
    recalls = [m['recall'] for m in region_metrics]
    f1s = [m['f1'] for m in region_metrics]
    accs = [m['classification_accuracy'] for m in region_metrics]

    # 绘制直方图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(precisions, bins=10, alpha=0.7)
    axes[0, 0].set_title('Precision Distribution')
    axes[0, 0].set_xlabel('Precision')
    axes[0, 0].set_ylabel('Count')

    axes[0, 1].hist(recalls, bins=10, alpha=0.7)
    axes[0, 1].set_title('Recall Distribution')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Count')

    axes[1, 0].hist(f1s, bins=10, alpha=0.7)
    axes[1, 0].set_title('F1 Score Distribution')
    axes[1, 0].set_xlabel('F1 Score')
    axes[1, 0].set_ylabel('Count')

    axes[1, 1].hist(accs, bins=10, alpha=0.7)
    axes[1, 1].set_title('Classification Accuracy Distribution')
    axes[1, 1].set_xlabel('Accuracy')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'region_detection_metrics.png'))

    # 绘制阅读顺序指标（如果有）
    if reading_order_metrics:
        plt.figure(figsize=(12, 5))

        accuracies = [m['accuracy'] for m in reading_order_metrics]
        bleu2s = [m['bleu2'] for m in reading_order_metrics]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(accuracies, bins=10, alpha=0.7)
        axes[0].set_title('Reading Order Accuracy Distribution')
        axes[0].set_xlabel('Accuracy')
        axes[0].set_ylabel('Count')

        axes[1].hist(bleu2s, bins=10, alpha=0.7)
        axes[1].set_title('BLEU-2 Score Distribution')
        axes[1].set_xlabel('BLEU-2')
        axes[1].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reading_order_metrics.png'))

        # 绘制混淆矩阵
        if len(region_metrics) > 0:
            # 收集所有的预测和真实标签
            all_gt_labels = []
            all_pred_labels = []

            # 这里不应该使用未定义的 results 变量
            # 而应该从 region_metrics 中获取必要的信息
            for metric in region_metrics:
                if 'matched_indices' in metric and 'gt_labels' in metric and 'pred_labels' in metric:
                    for gt_idx, pred_idx in metric['matched_indices']:
                        all_gt_labels.append(metric['gt_labels'][gt_idx])
                        all_pred_labels.append(metric['pred_labels'][pred_idx])

            # 如果收集到了标签数据，则绘制混淆矩阵
            if all_gt_labels and all_pred_labels:
                from sklearn.metrics import confusion_matrix
                import seaborn as sns

                # 类别标签
                class_names = ['Page', 'Page Number', 'Main Text', 'Note', 'Data', 'Table']

                # 创建混淆矩阵
                cm = confusion_matrix(all_gt_labels, all_pred_labels, labels=list(range(len(class_names))))

                # 绘制混淆矩阵
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Ground Truth')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

    plt.close('all')


if __name__ == '__main__':
    main()