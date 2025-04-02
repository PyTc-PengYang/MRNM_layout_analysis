import os
import argparse
import xml.etree.ElementTree as ET
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Convert data to the required format')
    parser.add_argument('--input_dir', type=str, required=True, help='input directory containing XML files and images')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='data split')
    return parser.parse_args()


def convert_page_xml(xml_path, image_dir, output_dir, split):
    # 解析XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {'ns0': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    # 获取图像文件名和尺寸
    page = root.find('ns0:Page', ns)
    image_filename = page.get('imageFilename')
    image_width = int(page.get('imageWidth'))
    image_height = int(page.get('imageHeight'))

    # 源图像路径和目标图像路径
    src_image_path = os.path.join(image_dir, image_filename)
    dst_image_path = os.path.join(output_dir, split, 'images', image_filename)

    # 确保目标目录存在
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)

    # 复制图像
    if os.path.exists(src_image_path):
        shutil.copy(src_image_path, dst_image_path)
    else:
        print(f"Warning: Image {src_image_path} not found")

    # 解析阅读顺序
    reading_order = []
    reading_order_elem = page.find('.//ns0:ReadingOrder/ns0:OrderedGroup', ns)
    if reading_order_elem is not None:
        for region_ref in reading_order_elem.findall('.//ns0:RegionRefIndexed', ns):
            index = int(region_ref.get('index'))
            region_id = region_ref.get('regionRef')
            reading_order.append((index, region_id))

        # 按索引排序
        reading_order.sort(key=lambda x: x[0])

    # 解析文本区域
    text_regions = []
    for region in page.findall('.//ns0:TextRegion', ns):
        region_id = region.get('id')
        region_type = region.get('type', '')

        # 提取区域坐标
        coords_elem = region.find('.//ns0:Coords', ns)
        points_str = coords_elem.get('points')
        coords = parse_points(points_str)

        # 提取文本行
        text_lines = []
        for line in region.findall('.//ns0:TextLine', ns):
            line_id = line.get('id')
            line_coords_elem = line.find('.//ns0:Coords', ns)
            line_points_str = line_coords_elem.get('points')
            line_coords = parse_points(line_points_str)

            # 提取基线
            baseline_elem = line.find('.//ns0:Baseline', ns)
            baseline = None
            if baseline_elem is not None:
                baseline_points_str = baseline_elem.get('points')
                baseline = parse_points(baseline_points_str)

            # 提取文本内容
            text_equiv = line.find('.//ns0:TextEquiv/ns0:Unicode', ns)
            text = text_equiv.text if text_equiv is not None and text_equiv.text else ""

            text_lines.append({
                'id': line_id,
                'coords': line_coords.tolist(),
                'baseline': baseline.tolist() if baseline is not None else None,
                'text': text
            })

        # 提取区域文本内容
        text_elem = region.find('.//ns0:Text', ns)
        region_text = ""
        if text_elem is not None and text_elem.text:
            region_text = text_elem.text

        # 计算边界框 (x1, y1, x2, y2)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

        text_regions.append({
            'id': region_id,
            'type': region_type,
            'bbox': bbox,
            'polygon': coords.tolist(),
            'text_lines': text_lines,
            'text': region_text
        })

    # 构建最终的标注
    annotation = {
        'image': {
            'filename': image_filename,
            'width': image_width,
            'height': image_height
        },
        'regions': text_regions,
        'reading_order': [(r_id, text_regions[i + 1]['id']) for i, (_, r_id) in enumerate(reading_order[:-1])]
    }

    # 保存标注为JSON
    base_filename = os.path.splitext(os.path.basename(xml_path))[0]
    with open(os.path.join(output_dir, split, f"{base_filename}.json"), 'w') as f:
        json.dump(annotation, f, indent=2)

    return annotation


def parse_points(points_str):
    point_pairs = points_str.split()
    coords = []

    for pair in point_pairs:
        x, y = pair.split(',')
        coords.append([int(x), int(y)])

    return np.array(coords)


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(os.path.join(args.output_dir, args.split), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.split, 'images'), exist_ok=True)

    # 遍历XML文件
    xml_files = [f for f in os.listdir(args.input_dir) if f.endswith('.xml') or f.endswith('.txt')]
    image_dir = os.path.join(args.input_dir, 'images')

    annotations = []

    for xml_file in tqdm(xml_files, desc=f"Converting {args.split} data"):
        xml_path = os.path.join(args.input_dir, xml_file)
        annotation = convert_page_xml(xml_path, image_dir, args.output_dir, args.split)
        annotations.append(annotation)

    print(f"Converted {len(annotations)} files to {os.path.join(args.output_dir, args.split)}")


if __name__ == '__main__':
    main()