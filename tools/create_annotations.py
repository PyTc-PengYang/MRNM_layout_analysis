import os
import argparse
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Create annotation files')
    parser.add_argument('--input_dir', type=str, required=True, help='input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory for annotations')
    parser.add_argument('--format', type=str, default='json', choices=['json', 'xml'], help='annotation format')
    return parser.parse_args()


def create_blank_annotation(image_path, output_path, format='json'):
    # 读取图像，获取尺寸
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    height, width, _ = image.shape
    image_filename = os.path.basename(image_path)

    if format == 'json':
        # 创建JSON格式标注
        annotation = {
            'image': {
                'filename': image_filename,
                'width': width,
                'height': height
            },
            'regions': [],
            'reading_order': []
        }

        # 保存为JSON文件
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)

    elif format == 'xml':
        # 创建PAGE XML格式标注
        root = ET.Element('PcGts')
        root.set('xmlns', 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15')

        metadata = ET.SubElement(root, 'Metadata')
        ET.SubElement(metadata, 'Creator').text = 'MRNM-Tool'

        page = ET.SubElement(root, 'Page')
        page.set('imageFilename', image_filename)
        page.set('imageWidth', str(width))
        page.set('imageHeight', str(height))

        # 创建打印空间
        printspace = ET.SubElement(page, 'PrintSpace')
        coords = ET.SubElement(printspace, 'Coords')
        coords.set('points', f'0,0 {width},0 {width},{height} 0,{height}')

        # 创建阅读顺序节点
        reading_order = ET.SubElement(page, 'ReadingOrder')
        ordered_group = ET.SubElement(reading_order, 'OrderedGroup')
        ordered_group.set('id', 'reading_order')
        ordered_group.set('caption', 'Regions reading order')

        # 保存为XML文件
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

    return os.path.basename(output_path)


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(args.input_dir) if
                   f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]

    # 为每个图像创建标注
    annotations = []

    for image_file in tqdm(image_files, desc="Creating annotations"):
        image_path = os.path.join(args.input_dir, image_file)

        # 确定输出文件路径
        base_name = os.path.splitext(image_file)[0]
        if args.format == 'json':
            output_path = os.path.join(args.output_dir, f"{base_name}.json")
        else:
            output_path = os.path.join(args.output_dir, f"{base_name}.xml")

        # 创建标注
        annotation_file = create_blank_annotation(image_path, output_path, args.format)
        if annotation_file:
            annotations.append(annotation_file)

    print(f"Created {len(annotations)} annotation files in {args.output_dir}")


if __name__ == '__main__':
    main()