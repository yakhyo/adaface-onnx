# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import argparse

from model import FaceRecognition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AdaFace Face Recognition - Compare two face images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image1", type=str, help="Path to first image")
    parser.add_argument("image2", type=str, help="Path to second image")
    parser.add_argument("--detector-path", type=str, default="weights/det_10g.onnx")
    parser.add_argument("--recognition-path", type=str, default="weights/adaface_ir_18.onnx")
    parser.add_argument("--threshold", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Initializing models...")
    fr = FaceRecognition(
        detector_path=args.detector_path,
        recognition_path=args.recognition_path,
    )

    print(f"Comparing: {args.image1} vs {args.image2}")
    similarity = fr.compare(args.image1, args.image2)

    if similarity is None:
        print("Error: Could not detect face in one or both images.")
        return

    match_status = "MATCH" if similarity >= args.threshold else "NO MATCH"
    print(f"\nSimilarity: {similarity:.4f}")
    print(f"Threshold:  {args.threshold}")
    print(f"Result:     {match_status}")


if __name__ == "__main__":
    main()
