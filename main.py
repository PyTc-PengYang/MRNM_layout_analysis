import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data_loader import LayoutDataset, get_transform
from models import MRNM
from scripts.train import train
from scripts.test import test
from scripts.visualize import visualize


def parse_args():
    parser = argparse.ArgumentParser(description='Document Layout Analysis with MRNM')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'visualize'],
                        help='program mode: train, test or visualize')
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='path to config file')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='path to model checkpoint (for test and visualize modes)')
    parser.add_argument('--data_dir', type=str, default='',
                        help='path to data directory (required for visualize mode)')
    parser.add_argument('--output', type=str, default='./output',
                        help='output directory')
    parser.add_argument('--resume', type=str, default='',
                        help='path to checkpoint to resume training (for train mode)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='number of samples to visualize (for visualize mode)')
    parser.add_argument('--show_reading_order', action='store_true',
                        help='visualize reading order (for visualize mode)')
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
    """递归合并字典"""
    for key, value in d2.items():
        if key in d1 and isinstance(value, dict) and isinstance(d1[key], dict):
            merge_dict(d1[key], value)
        else:
            d1[key] = value


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)

    if args.mode == 'train':
        print("Starting training...")
        train(config)

    elif args.mode == 'test':
        if not args.checkpoint:
            raise ValueError("Checkpoint path is required for test mode")

        print("Starting testing...")
        test(config, args.checkpoint, args.output)

    elif args.mode == 'visualize':
        if not args.checkpoint:
            raise ValueError("Checkpoint path is required for visualize mode")
        if not args.data_dir:
            raise ValueError("Data directory is required for visualize mode")

        print("Starting visualization...")
        visualize(
            config,
            args.checkpoint,
            args.data_dir,
            args.output,
            args.num_samples,
            args.show_reading_order
        )


if __name__ == '__main__':
    main()