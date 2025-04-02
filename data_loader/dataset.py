import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from .page_data import PageData


class LayoutDataset(Dataset):
    def __init__(self, data_dir, xml_list, image_dir, transform=None):
        self.data_dir = data_dir
        self.xml_paths = [os.path.join(data_dir, xml) for xml in xml_list]
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        self._load_samples()

    def _load_samples(self):
        for xml_path in self.xml_paths:
            page_data = PageData(xml_path, self.image_dir)
            self.samples.append(page_data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        page_data = self.samples[idx]

        # 加载图像
        image = page_data.load_image()

        # 创建区域标签和边界框
        regions = page_data.text_regions
        boxes = []
        labels = []
        texts = []

        # 定义区域类型映射
        region_type_map = {
            'Main Text': 1,
            'Page Number': 2,
            'Note': 3,
            'Data': 4,
            'Title': 5,
        }

        for region in regions:
            # 计算边界框 (左上角x, 左上角y, 右下角x, 右下角y)
            coords = region.coords
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)

            boxes.append([x_min, y_min, x_max, y_max])

            # 获取区域类型标签
            region_type = region.type.strip("'")
            label = region_type_map.get(region_type, 0)  # 默认为0（背景）
            labels.append(label)

            # 收集区域文本
            region_text = ""
            for line in region.text_lines:
                if line['text']:
                    region_text += line['text'] + " "

            texts.append(region_text.strip())

        # 获取阅读顺序关系
        reading_order = []
        ordered_regions = page_data.get_ordered_regions()
        for i in range(len(ordered_regions) - 1):
            current_idx = regions.index(ordered_regions[i])
            next_idx = regions.index(ordered_regions[i + 1])
            reading_order.append((current_idx, next_idx))

        # 转换为张量
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # 应用数据增强
        if self.transform:
            image, boxes = self.transform(image, boxes)

        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'texts': texts,
            'reading_order': reading_order,
            'image_id': os.path.basename(page_data.image_path)
        }