import xml.etree.ElementTree as ET
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class TextRegion:
    id: str
    type: str
    coords: np.ndarray  # 坐标点数组
    text_lines: List[Dict]  # 文本行
    text: str  # 区域文本内容


class PageData:
    def __init__(self, xml_path, image_dir):
        self.xml_path = xml_path
        self.image_dir = image_dir
        self.text_regions = []
        self.image = None
        self.reading_order = []
        self.parse_xml()

    def parse_xml(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        ns = {'ns0': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

        # 获取图像文件名和尺寸
        page = root.find('ns0:Page', ns)
        image_filename = page.get('imageFilename')
        self.image_width = int(page.get('imageWidth'))
        self.image_height = int(page.get('imageHeight'))

        # 加载图像
        self.image_path = f"{self.image_dir}/{image_filename}"

        # 解析阅读顺序
        reading_order = page.find('.//ns0:ReadingOrder/ns0:OrderedGroup', ns)
        if reading_order is not None:
            for region_ref in reading_order.findall('.//ns0:RegionRefIndexed', ns):
                index = int(region_ref.get('index'))
                region_id = region_ref.get('regionRef')
                self.reading_order.append((index, region_id))

            # 按索引排序
            self.reading_order.sort(key=lambda x: x[0])

        # 解析文本区域
        for region in page.findall('.//ns0:TextRegion', ns):
            region_id = region.get('id')
            region_type = region.get('type', '')

            # 提取区域坐标
            coords_elem = region.find('.//ns0:Coords', ns)
            points_str = coords_elem.get('points')
            coords = self._parse_points(points_str)

            # 提取文本行
            text_lines = []
            for line in region.findall('.//ns0:TextLine', ns):
                line_id = line.get('id')
                line_coords_elem = line.find('.//ns0:Coords', ns)
                line_points_str = line_coords_elem.get('points')
                line_coords = self._parse_points(line_points_str)

                # 提取基线
                baseline_elem = line.find('.//ns0:Baseline', ns)
                baseline = None
                if baseline_elem is not None:
                    baseline_points_str = baseline_elem.get('points')
                    baseline = self._parse_points(baseline_points_str)

                # 提取文本内容
                text_equiv = line.find('.//ns0:TextEquiv/ns0:Unicode', ns)
                text = text_equiv.text if text_equiv is not None and text_equiv.text else ""

                text_lines.append({
                    'id': line_id,
                    'coords': line_coords,
                    'baseline': baseline,
                    'text': text
                })

            # 提取区域文本内容
            text_elem = region.find('.//ns0:Text', ns)
            region_text = ""
            if text_elem is not None and text_elem.text:
                region_text = text_elem.text

            text_region = TextRegion(
                id=region_id,
                type=region_type,
                coords=coords,
                text_lines=text_lines,
                text=region_text
            )

            self.text_regions.append(text_region)

    def _parse_points(self, points_str):
        point_pairs = points_str.split()
        coords = []

        for pair in point_pairs:
            x, y = pair.split(',')
            coords.append([int(x), int(y)])

        return np.array(coords)

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"图像文件未找到: {self.image_path}")
        return self.image

    def get_text_region_by_id(self, region_id):
        for region in self.text_regions:
            if region.id == region_id:
                return region
        return None

    def get_ordered_regions(self):
        ordered_regions = []
        for _, region_id in self.reading_order:
            region = self.get_text_region_by_id(region_id)
            if region:
                ordered_regions.append(region)
        return ordered_regions