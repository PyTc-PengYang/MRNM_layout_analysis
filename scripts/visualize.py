import os
import argparse
import yaml
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from data_loader import LayoutDataset, get_transform
from models import MRNM


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize MRNM model predictions')
    parser.add_argument('--config', type=str, default='configs/model.yaml', help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='path to test data directory')
    parser.add_argument('--output', type=str, default='./visualizations', help='output directory')
    parser.add_argument('--num_samples', type=int, default=10, help='number of samples to visualize')
    parser.add_argument('--show_reading_order', action='store_true', help='visualize reading order')
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


def visualize(config, checkpoint_path, data_dir, output_dir, num_samples, show_reading_order):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    xml_files = os.listdir(data_dir)
    if len(xml_files) > num_samples:
        xml_files = xml_files[:num_samples]

    dataset = LayoutDataset(
        data_dir=data_dir,
        xml_list=xml_files,
        image_dir=os.path.join(data_dir, 'images'),
        transform=get_transform(train=False)
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

    # 类别颜色映射
    colors = [
        (255, 0, 0),  # 红色 - 页面
        (0, 255, 0),  # 绿色 - 页码
        (0, 0, 255),  # 蓝色 - 正文
        (255, 255, 0),  # 黄色 - 注释
        (255, 0, 255),  # 洋红色 - 数据
        (0, 255, 255)  # 青色 - 表格
    ]

    # 类别名称
    class_names = ['Page', 'Page Number', 'Main Text', 'Note', 'Data', 'Table']

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Visualizing"):
            sample = dataset[i]

            # 准备数据
            image = sample['image'].unsqueeze(0).to(device)
            gt_boxes = sample['boxes'].to(device)
            gt_labels = sample['labels'].to(device)
            texts = sample['texts']
            image_id = sample['image_id']

            # 前向传播
            outputs = model(image, None, [texts])

            # 提取预测结果
            pred_boxes = outputs['proposals'][0]
            pred_scores = outputs['region_scores'][0]
            _, pred_labels = torch.max(pred_scores, dim=1)

            # 预测的阅读顺序
            pred_reading_order = outputs['reading_orders'][0] if 'reading_orders' in outputs else []

            # 转换回原始图像
            # 从tensor转换回numpy图像
            img = image[0].cpu().numpy().transpose(1, 2, 0)
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)

            # 创建可视化图像
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # 绘制ground truth
            ax1.imshow(img)
            ax1.set_title('Ground Truth')
            for j, (box, label) in enumerate(zip(gt_boxes.cpu().numpy(), gt_labels.cpu().numpy())):
                x1, y1, x2, y2 = box
                color = colors[label % len(colors)]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=np.array(color) / 255,
                    facecolor='none'
                )
                ax1.add_patch(rect)
                ax1.text(
                    x1, y1 - 5, f"{class_names[label]}",
                    color='white', fontsize=8,
                    bbox=dict(facecolor=np.array(color) / 255, alpha=0.5)
                )

            # 绘制预测结果
            ax2.imshow(img)
            ax2.set_title('Predictions')
            for j, (box, label, score) in enumerate(zip(
                    pred_boxes.cpu().numpy(),
                    pred_labels.cpu().numpy(),
                    pred_scores.softmax(dim=1).max(dim=1)[0].cpu().numpy()
            )):
                x1, y1, x2, y2 = box
                color = colors[label % len(colors)]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=np.array(color) / 255,
                    facecolor='none'
                )
                ax2.add_patch(rect)
                ax2.text(
                    x1, y1 - 5, f"{class_names[label]} ({score:.2f})",
                    color='white', fontsize=8,
                    bbox=dict(facecolor=np.array(color) / 255, alpha=0.5)
                )

                # 在框内添加序号
                ax2.text(
                    x1 + 5, y1 + 20, f"{j}",
                    color='white', fontsize=12, weight='bold',
                    bbox=dict(facecolor='black', alpha=0.7)
                )

            # 绘制阅读顺序
            if show_reading_order and pred_reading_order:
                for j, (src, dst) in enumerate(pred_reading_order):
                    src_box = pred_boxes[src].cpu().numpy()
                    dst_box = pred_boxes[dst].cpu().numpy()

                    src_center = [(src_box[0] + src_box[2]) / 2, (src_box[1] + src_box[3]) / 2]
                    dst_center = [(dst_box[0] + dst_box[2]) / 2, (dst_box[1] + dst_box[3]) / 2]

                    ax2.arrow(
                        src_center[0], src_center[1],
                        dst_center[0] - src_center[0], dst_center[1] - src_center[1],
                        head_width=10, head_length=10, fc='white', ec='black', alpha=0.7
                    )

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{image_id.split('.')[0]}_viz.png"))
            plt.close()

            # 另外创建一个单独的阅读顺序可视化
            if show_reading_order and pred_reading_order:
                # 创建新的图像
                reading_order_img = img.copy()
                pil_img = Image.fromarray(reading_order_img)
                draw = ImageDraw.Draw(pil_img)

                # 尝试加载字体，如果失败则使用默认字体
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()

                # 绘制区域框和编号
                for j, box in enumerate(pred_boxes.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)
                    label = pred_labels[j].item()
                    color = colors[label % len(colors)]

                    # 绘制矩形框
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                    # 绘制编号
                    text_position = (x1 + 5, y1 + 5)
                    draw.text(text_position, str(j), fill=(255, 255, 255), font=font)

                # 绘制阅读顺序连线和箭头
                for j, (src, dst) in enumerate(pred_reading_order):
                    src_box = pred_boxes[src].cpu().numpy()
                    dst_box = pred_boxes[dst].cpu().numpy()

                    src_center = (int((src_box[0] + src_box[2]) / 2), int((src_box[1] + src_box[3]) / 2))
                    dst_center = (int((dst_box[0] + dst_box[2]) / 2), int((dst_box[1] + dst_box[3]) / 2))

                    # 绘制连线
                    draw.line([src_center, dst_center], fill=(255, 255, 255), width=2)

                    # 绘制箭头（简化版）
                    arrow_length = 20
                    dx = dst_center[0] - src_center[0]
                    dy = dst_center[1] - src_center[1]
                    length = (dx ** 2 + dy ** 2) ** 0.5
                    if length > 0:
                        dx, dy = dx / length, dy / length
                        # 计算箭头端点
                        arrow_x = int(dst_center[0] - arrow_length * dx)
                        arrow_y = int(dst_center[1] - arrow_length * dy)
                        # 计算箭头两侧的点
                        arrow_dx = int(arrow_length * dy * 0.5)
                        arrow_dy = int(arrow_length * dx * 0.5)

                        # 绘制箭头头部
                        draw.line([
                            (arrow_x + arrow_dx, arrow_y - arrow_dy),
                            dst_center,
                            (arrow_x - arrow_dx, arrow_y + arrow_dy)
                        ], fill=(255, 255, 255), width=2)

                # 保存阅读顺序图像
                pil_img.save(os.path.join(output_dir, f"{image_id.split('.')[0]}_reading_order.png"))

        print(f"Visualization completed! Results saved to {output_dir}")

        if __name__ == '__main__':
            args = parse_args()
            config = load_config(args.config)
            visualize(
                config,
                args.checkpoint,
                args.data_dir,
                args.output,
                args.num_samples,
                args.show_reading_order
            )