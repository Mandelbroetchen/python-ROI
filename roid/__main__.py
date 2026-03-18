import argparse
from PIL import Image
import numpy as np
from roid.imset import Imset
from roit.roit import Roit
from roid.roid import Roid

def parse_args():
    parser = argparse.ArgumentParser(description="ROI-database Image Transformation")
    parser.add_argument("-s", "--source", type=str, help="Path to source folder")
    parser.add_argument("-t", "--target", default=None, type=str, help="Path to target folder (optional)")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    
    roit = Roit()

    roid = Roid(args.source, args.target, roit)

    roid.transform()

    roid.target.save()
    
    