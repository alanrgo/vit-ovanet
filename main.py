import argparse
import torch
import os

from default import _C as cfg


# RANK is Global rank of the current process across all machines
# Not available, so I commented the code

# RANK = int(os.environ['RANK']) if torch.cuda.device_count() > 1 else 0
# LOCAL_RANK = int(os.environ['LOCAL_RANK']) if torch.cuda.device_count() > 1 else 0
# WORLD_SIZE = int(os.environ['WORLD_SIZE']) if torch.cuda.device_count() > 1 else 1

RANK = 0 
LOCAL_RANK = 0
WORLD_SIZE = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Pytorch Implementation of "Vit-ovanet: Open set segmentation for phenology images"''')
    parser.add_argument('--config-file', default='', help='path to config gile', type=str)
    parser.add_argument('opts', default=None, help='modify config using the command-line', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts) # example: python3 main.py --config-file configs/vit_base.yaml MODEL.NAME vit_base MODEL.BIAS True
    cfg.freeze() # Makes it read-only

    OUTPUT_DIR = cfg.MISC.OUTPUT_DIR + '_' + cfg.MISC.SUFFIX if cfg.MISC.SUFFIX else cfg.MISC.OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
