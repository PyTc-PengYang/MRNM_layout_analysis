import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import time

from data_loader import LayoutDataset, get_transform
from models import MRNM
from utils import ContrastiveLoss, RegularizationLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train MRNM model')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='path to config file')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config):
    # 创建输出目录
    os.makedirs(config['output']['save_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_dataset = LayoutDataset(
        data_dir=config['dataset']['train_dir'],
        xml_list=os.listdir(config['dataset']['train_dir']),
        image_dir=os.path.join(config['dataset']['train_dir'], 'images'),
        transform=get_transform(train=True)
    )

    val_dataset = LayoutDataset(
        data_dir=config['dataset']['val_dir'],
        xml_list=os.listdir(config['dataset']['val_dir']),
        image_dir=os.path.join(config['dataset']['val_dir'], 'images'),
        transform=get_transform(train=False)
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['eval']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 创建模型
    model = MRNM(
        num_classes=config['dataset']['num_classes'],
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)

    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )

    # 创建学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['train']['lr_step_size'],
        gamma=config['train']['lr_gamma']
    )

    # 创建损失函数
    contrastive_loss = ContrastiveLoss().to(device)
    regularization_loss = RegularizationLoss(weight=config['train']['loss_weights']['reg_loss']).to(device)

    # 创建TensorBoard日志
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config['output']['log_dir'], f'run_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)

    # 训练循环
    start_epoch = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0

    # 恢复训练
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # 开始训练
    print("Start training...")
    for epoch in range(start_epoch, config['train']['num_epochs']):
        # 训练一个epoch
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, contrastive_loss, regularization_loss,
            device, epoch, config
        )

        # 验证
        val_loss, val_metrics = validate(
            model, val_loader, contrastive_loss, regularization_loss,
            device, epoch, config
        )

        # 更新学习率
        lr_scheduler.step()

        # 记录日志
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        for metric, value in train_metrics.items():
            writer.add_scalar(f'Metrics/train_{metric}', value, epoch)

        for metric, value in val_metrics.items():
            writer.add_scalar(f'Metrics/val_{metric}', value, epoch)

        # 保存最佳模型
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }, is_best, config['output']['checkpoint_dir'])

        # 早停
        if is_best:
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config['train']['early_stopping']:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()
    print("Training completed!")


def train_one_epoch(model, data_loader, optimizer, contrastive_loss, regularization_loss, device, epoch, config):
    model.train()
    total_loss = 0.0
    metrics = {
        'region_accuracy': 0.0,
        'relation_accuracy': 0.0,
    }

    start_time = time.time()
    for i, batch in enumerate(data_loader):
        # 准备数据
        images = batch['image'].to(device)
        boxes = [box.to(device) for box in batch['boxes']]
        labels = [label.to(device) for label in batch['labels']]
        texts = batch['texts']
        reading_order = batch['reading_order'] if 'reading_order' in batch else None

        # 前向传播
        outputs = model(images, boxes, texts, reading_order)

        # 计算损失
        losses = model.compute_losses({
            'region_scores': outputs['region_scores'],
            'relation_probs': outputs['relation_probs'] if 'relation_probs' in outputs else None,
            'visual_embeds': outputs.get('visual_embeds', None),
            'text_embeds': outputs.get('text_embeds', None),
            'spatial_embeds': outputs.get('spatial_embeds', None),
        }, {
            'labels': labels,
            'reading_order': reading_order
        })

        # 总损失
        loss = losses['total_loss']

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()

        # 计算指标
        with torch.no_grad():
            # 区域分类准确率
            region_preds = []
            for i, scores in enumerate(outputs['region_scores']):
                if scores is not None:
                    _, preds = torch.max(scores, dim=1)
                    correct = torch.sum(preds == labels[i]).item()
                    total = labels[i].size(0)
                    region_preds.append(correct / total)

            if region_preds:
                metrics['region_accuracy'] += sum(region_preds) / len(region_preds)

            # 关系预测准确率
            if 'relation_probs' in outputs and 'reading_orders' in outputs:
                relation_accuracy = 0.0
                for i, (pred_order, gt_order) in enumerate(zip(outputs['reading_orders'], reading_order)):
                    if pred_order and gt_order:
                        # 计算阅读顺序准确率
                        correct = 0
                        for j, (src, dst) in enumerate(gt_order):
                            if j < len(pred_order) and pred_order[j] == (src, dst):
                                correct += 1

                        relation_accuracy += correct / len(gt_order) if gt_order else 0

                if reading_order:
                    metrics['relation_accuracy'] += relation_accuracy / len(reading_order)

        # 打印进度
        if (i + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {i + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")

    # 计算平均值
    avg_loss = total_loss / len(data_loader)
    for metric in metrics:
        metrics[metric] /= len(data_loader)

    # 打印结果
    elapsed = time.time() - start_time
    print(f"Epoch {epoch + 1} completed in {elapsed:.2f}s, Avg Loss: {avg_loss:.4f}, "
          f"Region Acc: {metrics['region_accuracy']:.4f}, Relation Acc: {metrics['relation_accuracy']:.4f}")

    return avg_loss, metrics


def validate(model, data_loader, contrastive_loss, regularization_loss, device, epoch, config):
    model.eval()
    total_loss = 0.0
    metrics = {
        'region_accuracy': 0.0,
        'relation_accuracy': 0.0,
        'iou': 0.0,
    }

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # 准备数据
            images = batch['image'].to(device)
            boxes = [box.to(device) for box in batch['boxes']]
            labels = [label.to(device) for label in batch['labels']]
            texts = batch['texts']
            reading_order = batch['reading_order'] if 'reading_order' in batch else None

            # 前向传播
            outputs = model(images, boxes, texts, reading_order)

            # 计算损失
            losses = model.compute_losses({
                'region_scores': outputs['region_scores'],
                'relation_probs': outputs['relation_probs'] if 'relation_probs' in outputs else None,
                'visual_embeds': outputs.get('visual_embeds', None),
                'text_embeds': outputs.get('text_embeds', None),
                'spatial_embeds': outputs.get('spatial_embeds', None),
            }, {
                'labels': labels,
                'reading_order': reading_order
            })

            # 总损失
            loss = losses['total_loss']
            total_loss += loss.item()

            # 计算指标
            # 区域分类准确率
            region_preds = []
            for i, scores in enumerate(outputs['region_scores']):
                if scores is not None:
                    _, preds = torch.max(scores, dim=1)
                    correct = torch.sum(preds == labels[i]).item()
                    total = labels[i].size(0)
                    region_preds.append(correct / total)

            if region_preds:
                metrics['region_accuracy'] += sum(region_preds) / len(region_preds)

            # 计算IoU
            iou_sum = 0.0
            iou_count = 0
            for i, (pred_boxes, gt_boxes) in enumerate(zip(outputs['proposals'], boxes)):
                if pred_boxes.size(0) > 0 and gt_boxes.size(0) > 0:
                    # 计算每个预测框与真实框的IoU
                    ious = box_iou(pred_boxes, gt_boxes)
                    # 取每个预测框的最大IoU
                    max_ious, _ = torch.max(ious, dim=1)
                    iou_sum += max_ious.mean().item()
                    iou_count += 1

            if iou_count > 0:
                metrics['iou'] += iou_sum / iou_count

            # 关系预测准确率
            if 'relation_probs' in outputs and 'reading_orders' in outputs:
                relation_accuracy = 0.0
                for i, (pred_order, gt_order) in enumerate(zip(outputs['reading_orders'], reading_order)):
                    if pred_order and gt_order:
                        # 计算阅读顺序准确率
                        correct = 0
                        for j, (src, dst) in enumerate(gt_order):
                            if j < len(pred_order) and pred_order[j] == (src, dst):
                                correct += 1

                        relation_accuracy += correct / len(gt_order) if gt_order else 0

                if reading_order:
                    metrics['relation_accuracy'] += relation_accuracy / len(reading_order)

    # 计算平均值
    avg_loss = total_loss / len(data_loader)
    for metric in metrics:
        metrics[metric] /= len(data_loader)

    # 打印结果
    print(f"Validation - Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}, "
          f"Region Acc: {metrics['region_accuracy']:.4f}, Relation Acc: {metrics['relation_accuracy']:.4f}, "
          f"IoU: {metrics['iou']:.4f}")

    return avg_loss, metrics


def save_checkpoint(state, is_best, checkpoint_dir):
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filename)


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


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
    set_seed(args.seed)
    train(config)