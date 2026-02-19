import argparse
from .roii import Roii
import json
# --- Command line interface ---
def parse_args():
    parser = argparse.ArgumentParser(description="ROI inference on images using BLIP2")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder with images")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate per image")
    parser.add_argument("--N", type=int, default=None, help="Max process number")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    roii = Roii()

    results = roii.process_folder(
        image_folder=args.image_folder,
        max_new_tokens=args.max_tokens,
        N = args.N
    )

    print(results)