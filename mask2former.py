import os
import cv2
import json
import torch
import argparse
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from Mask2Former.mask2former import add_maskformer2_config
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    if args.model == 'COCO':
        config_file = "Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml"
        weight_path = "weights/mask2former/coco/panoptic/model_final_9d7f02.pkl"
    elif args.model == 'ADE20K':
        config_file = "Mask2Former/configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_160k.yaml"
        weight_path = "weights/mask2former/ade20k/panoptic/model_final_5c90d4.pkl"
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHTS", weight_path])
    cfg.freeze()
    
    return cfg


def run_inference(predictor, image_path: str):
    image = read_image(image_path, format="BGR")
    predictions = predictor(image)
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    return image, panoptic_seg, segments_info


def visualize_and_save(image, panoptic_seg, segments_info, metadata, save_path: str):
    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to(torch.device("cpu")), segments_info
    )
    vis_img = vis_output.get_image()[:, :, ::-1]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis_img)
    print(f"Saved visualized result to {save_path}")


def save_results(panoptic_seg, segments_info, output_json_path: str):
    panoptic_seg_np = panoptic_seg.to(torch.device("cpu")).numpy().tolist()
    results = {
        "panoptic_seg": panoptic_seg_np,
        "segments_info": segments_info
    }
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved panoptic_seg and segments_info to {output_json_path}")


def get_parser():
    parser = argparse.ArgumentParser(description="Mask2Former inference script")
    parser.add_argument(
        "--model",
        default="COCO",
        choices=["COCO", "ADE20K"],
        help="Model to use for inference"
    )
    parser.add_argument(
        "--input",
        default="data/example.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--output-dir",
        default="results/mask2former",
        help="Directory to save results"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Whether to save visualization"
    )
    return parser


def main():
    args = get_parser().parse_args()
    
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")

    file_name = os.path.splitext(os.path.basename(args.input))[0]
    
    image, panoptic_seg, segments_info = run_inference(predictor, args.input)

    if args.visualize:
        output_img_path = os.path.join(args.output_dir, f"{file_name}.png")
        visualize_and_save(image, panoptic_seg, segments_info, metadata, output_img_path)

    output_json_path = os.path.join(args.output_dir, f"{file_name}.json")
    save_results(panoptic_seg, segments_info, output_json_path)


if __name__ == "__main__":
    main()
