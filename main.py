import argparse
import builtins
import torch
import torch.distributed as dist
import os

import datasets
from default import _C as cfg
from utils.logger import setup_logger

def main():
    device_id = LOCAL_RANK
    torch.cuda.set_device(device_id)
    torch.backends.cudnn.benchmark = True

    # TF32 (TensorFloat-32) is a mathematical format introduced by NVIDIA 
    # for accelerating deep learning computations on Ampere GPUs and newer 
    # (RTX 30xx series, A100, etc.).
    torch.backends.cuda.matmul.allow_tf32 = cfg.ENGINE.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = cfg.ENGINE.ALLOW_TF32

    logger.info(f'set cuda device = {device_id}')

    if WORLD_SIZE > 1:
        logger.info(f'initializing distrubuted training environment')
        dist.init_process_group(backend='nccl')

    logger.info(f'initializing data loader')
    train_loader, eval_loader = initialize_loader()

def initialize_loader():
    train_dataset = datasets.DefaultDataset(cfg.DATASET.TRAIN)
    val_dataset = datasets.DefaultDataset(cfg.DATASET.EVAL)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg. ENGINE.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.ENGINE.NUM_WORKERS,
        pin_memory=cfg.ENGINE.PIN_MEMORY
    )

    eval_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.ENGINE.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.ENGINE.NUM_WORKERS,
        pin_memory=cfg.ENGINE.PIN_MEMORY
    )

    return train_loader, eval_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Pytorch Implementation of "Vit-ovanet: Open set segmentation for phenology images"''')
    parser.add_argument('--config-file', default='', help='path to config gile', type=str)
    parser.add_argument('opts', default=None, help='modify config using the command-line', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts) # example: python3 main.py --config-file configs/vit_base.yaml MODEL.NAME vit_base MODEL.BIAS True
    cfg.freeze() # Makes it read-only

    # RANK is Global rank of the current process across all machines
    # Not available, so I commented the code

    # RANK = int(os.environ['RANK']) if torch.cuda.device_count() > 1 else 0
    # LOCAL_RANK = int(os.environ['LOCAL_RANK']) if torch.cuda.device_count() > 1 else 0
    # WORLD_SIZE = int(os.environ['WORLD_SIZE']) if torch.cuda.device_count() > 1 else 1

    RANK = 0 
    LOCAL_RANK = 0
    WORLD_SIZE = 1

    OUTPUT_DIR = cfg.MISC.OUTPUT_DIR + '_' + cfg.MISC.SUFFIX if cfg.MISC.SUFFIX else cfg.MISC.OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = setup_logger(OUTPUT_DIR, LOCAL_RANK, cfg.MODEL.NAME)
    with open(os.path.join(OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    if LOCAL_RANK != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    main()