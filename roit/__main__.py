from .roit import Roit
import argparse
from PIL import Image
import numpy as np
from roid.imset import Imset

def parse_args():
    parser = argparse.ArgumentParser(description="ROI-based image transformation using IP-Adapter.")
    parser.add_argument("-i", "--input", type=str, help="Path to input image/folder")
    parser.add_argument("-o", "--output", default=None, type=str, help="Path to save output image/folder")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    roit = Roit()

    imset_in = Imset(args.input)

    imset_out = roit.transform_imset(imset_in)

    if args.output is not None:
        imset_out.root = args.output
    
    imset_out.save()
    
    