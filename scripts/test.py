import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

from data_loader import LayoutDataset, get_transform
from models import MRNM
from utils import evaluate_region_detection, evaluate_reading_order


def parse_args():
    parser = argparse.ArgumentParser(description='Test MRNM model')
    parser.add_argument('--config', type=str, default='configs/model.yaml', help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--output', type=str, default='./results', help='output directory')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 检查是否继承其他配置
    if 'inherit' in config:
        parent_config_path = os.path.join(os.path.dirname(config_path), config['inherit'])
        parent_config = load_config(parent_config_path)

        # 合并配置
        for key, value in config.items():
            if key != 'inherit':
                if key in parent_config and isinstance(value, dict) and isinstance(parent_config[key], dict):
                    # 递归合并字典
                    merge_dict(parent_config[key], value)
                else:
                    parent_config[key] = value

        return parent_config

    return config


def merge_dict(d1, d2):
    for key, value in d2.items():
        if key in d1 and isinstance(value, dict) and isinstance(d1[key], dict):
            merge_dict(d1[key], value)
        else:
            d1[key] = value


def test(config, checkpoint_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试数据集
    test_dataset = LayoutDataset(
        data_dir=config['dataset']['test_dir'],
        xml_list=os.listdir(config['dataset']['test_dir']),
        image_dir=os.path.join(config['dataset']['test_dir'], 'images'),
        transform=get_transform(train=False)
    )

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['eval']['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 创建模型
    model = MRNM(
        num_classes=config['dataset']['num_classes'],
        pretrained=False
    )

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    # 测试结果
    results = []

    # 评估指标
    region_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'classification_accuracy': 0.0
    }
    reading_order_accuracy = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # 准备数据
            images = batch['image'].to(device)
            gt_boxes = [b.to(device) for b in batch['boxes']]
            gt_labels = [l.to(device) for l in batch['labels']]
            texts = batch['texts']
            gt_reading_order = batch['reading_order'] if 'reading_order' in batch else None
            image_ids = batch['image_id']

            # 前向传播
            outputs = model(images, None, texts)

            # 处理每个样本
            for i in range(images.size(0)):
                # 提取预测结果
                pred_boxes = outputs['proposals'][i]
                pred_scores = outputs['region_scores'][i]
                _, pred_labels = torch.max(pred_scores, dim=1)

                # 预测的阅读顺序
                pred_reading_order = outputs['reading_orders'][i] if 'reading_orders' in outputs else []

                # 评估区域检测性能
                metrics = evaluate_region_detection(
                    gt_boxes[i].cpu().numpy(),
                    pred_boxes.cpu().numpy(),
                    gt_labels[i].cpu().numpy(),
                    pred_labels.cpu().numpy(),
                    iou_threshold=config['eval']['iou_threshold']
                )

                for metric, value in metrics.items():
                    region_metrics[metric] += value

                # 评估阅读顺序性能
                if gt_reading_order and gt_reading_order[i]:
                    order_acc = evaluate_reading_order(
                        gt_reading_order[i],
                        pred_reading_order
                    )
                    reading_order_accuracy += order_acc

                # 保存结果
                result = {
                    'image_id': image_ids[i],
                    'boxes': pred_boxes.cpu().numpy().tolist(),
                    'labels': pred_labels.cpu().numpy().tolist(),
                    'scores': pred_scores.softmax(dim=1).max(dim=1)[0].cpu().numpy().tolist(),
                    'reading_order': pred_reading_order,
                    'metrics': {
                        'region': {metric: value for metric, value in metrics.items()},
                        'reading_order': order_acc if gt_reading_order and gt_reading_order[i] else None
                    }
                }

                results.append(result)

    # 计算平均值
    num_samples = len(test_loader.dataset)
    for metric in region_metrics:
        region_metrics[metric] /= num_samples

    reading_order_accuracy /= num_samples

    # 总体指标
    overall_metrics = {
        'region': region_metrics,
        'reading_order': {'accuracy': reading_order_accuracy}
    }

    # 保存结果
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=2)

    # 打印结果
    print("Test completed!")
    print(f"Region Detection - Precision: {region_metrics['precision']:.4f}, Recall: {region_metrics['recall']:.4f}, "
          f"F1: {region_metrics['f1']:.4f}, Classification Accuracy: {region_metrics['classification_accuracy']:.4f}")
    print(f"Reading Order - Accuracy: {reading_order_accuracy:.4f}")


def collate_fn(batch):
    images = []
    boxes = []
    labels = []
    texts = []
    reading_orders = []
    image_ids = []

    for item in batch:
        images.append(item['image'])
        boxes.append(item['boxes'])
        labels.append(item['labels'])
        texts.append(item['texts'])
        reading_orders.append(item.get('reading_order', []))
        image_ids.append(item.get('image_id', ''))

    # 堆叠图像
    images = torch.stack(images, dim=0)

    return {
        'image': images,
        'boxes': boxes,
        'labels': labels,
        'texts': texts,
        'reading_order': reading_orders,
        'image_id': image_ids
    }


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    test(config, args.checkpoint, args.output)