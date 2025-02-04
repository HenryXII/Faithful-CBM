import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# import sys
# sys.path.append('~private/VLG-CBM')

import data.utils as data_utils
import groundingdino.datasets.transforms as T
from data.utils import get_data
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, file_name, box_list, label_list, img_path=None):
    json_data = [
        {
            "img_path": img_path,
        }
    ]
    for label, box in zip(label_list, box_list):
        name, logit = label.split("(")
        logit = logit[:-1]  # the last is ')'
        json_data.append(
            {
                "label": name,
                "logit": float(logit),
                "box": box.numpy().tolist(),
            }
        )
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(json_data, f)


def get_phrases_from_posmap(
    logits: torch.Tensor,
    text_threshold: float,
    tokenized: Dict,
    tokenizer: AutoTokenizer,
):
    assert isinstance(logits, torch.Tensor), "posmap must be torch.Tensor"
    if logits.dim() == 1:
        logits = logits[1:-1]
        tokenized = np.array(tokenized["input_ids"][1:-1])
        texts = np.split(
            tokenized, np.argwhere(tokenized == 1012).flatten()
        )  # 1012 is the " . " token which is a separator for GroundingDINO
        logits = np.split(logits, np.argwhere(tokenized == 1012).flatten())
        filtered_texts = [
            (tokenizer.decode(text).replace(".", "").strip(), logit.max())
            for text, logit in zip(texts, logits)
            if logit.max() > text_threshold
        ]
        return filtered_texts
    else:
        raise NotImplementedError("posmap must be 1-dim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument(
        "--config",
        type=str,
        default="home/jix049/private/VLG-CBM/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        help="path to config file",
    )
    parser.add_argument(
        "--grounded_checkpoint",
        type=str,
        default="home/jix049/private/VLG-CBM/GroundingDINO/groundingdino_swinb_cogcoor.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="outputs",
        help="output directory",
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--start_class_idx", type=int, default=None, help="start index (inclusive)")
    parser.add_argument("--end_class_idx", type=int, default=None, help="end index (exclusive)")
    parser.add_argument("--batch_size", type=int, default=48, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--dataset", type=str, default="cifar10_train", help="dataset name")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--save_image", action="store_true", help="save image")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    batch_size = args.batch_size
    num_workers = args.num_workers
    dataset_name = args.dataset
    device = args.device
    save_image = args.save_image
    output_dir = f"{args.output_dir}/{dataset_name}"

    # load classes
    cls_file = data_utils.LABEL_FILES[args.dataset.split("_")[0]]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")

    # load per class concepts
    per_class_concepts_file = f"concept_files/{dataset_name.split('_')[0]}_per_class.json"
    with open(per_class_concepts_file, "r") as f:
        per_class_concepts = json.load(f)

    # load PIL dataset to obtain original images
    pil_data = data_utils.get_data(dataset_name, preprocess=None)

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # prepare dataset
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = get_data(dataset_name, preprocess=lambda x: transform(x, None)[0])

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    model = model.to(device)
    tokenlizer = model.tokenizer

    # setup prompt
    for class_idx, class_name in enumerate(classes):
        # check if class_idx is in range of start and end
        if args.start_class_idx is not None and class_idx < args.start_class_idx:
            continue
        if args.end_class_idx is not None and class_idx >= args.end_class_idx:
            continue

        print(f"Running on class Index: {class_idx}, class name: {class_name}")
        per_class_concept = per_class_concepts[class_name]

        # setup prompt
        # add class name since it leads to better bounding boxes
        # we remove it during model training in the Concept dataset class
        text_prompt = data_utils.format_concept(class_name) + " . "
        for concept in per_class_concept:
            text_prompt = text_prompt + f"{data_utils.format_concept(concept)} . "
        text_prompt = text_prompt.strip()
        tokenized = tokenlizer(text_prompt)
        captions = [text_prompt] * batch_size
        print(f"Prompt for class {class_name}: {text_prompt}")

        # only load images with class_idx
        dataset_subset = torch.utils.data.Subset(dataset, np.where(np.array(dataset.targets) == class_idx)[0])
        dataloader = torch.utils.data.DataLoader(
            dataset_subset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )
        print(f"Number of images in class {class_name}: {len(dataset_subset)}")

        # run model
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images, captions=captions[: images.shape[0]])

            # get output
            logits = outputs["pred_logits"].sigmoid()  # (batch_size, nq, 256)
            boxes = outputs["pred_boxes"]  # (batch_size, nq, 4)

            # filter output
            filt_mask = logits.max(dim=2)[0] > box_threshold

            for image_idx in range(logits.shape[0]):
                processed_images = batch_idx * batch_size + image_idx
                global_idx = dataset_subset.indices[processed_images]
                pred_phrases = []
                boxes_final = []

                # get data for image
                logits_filt = logits[image_idx][filt_mask[image_idx]].clone().cpu()
                boxes_filt = boxes[image_idx][filt_mask[image_idx]].clone().cpu()
                for logit, box in zip(logits_filt, boxes_filt):
                    output__ = get_phrases_from_posmap(logit, text_threshold, tokenized, tokenlizer)
                    for i in range(len(output__)):
                        boxes_final.append(box)
                        pred_phrases.append(output__[i][0] + f"({str(output__[i][1].item())[:4]})")

                if len(boxes_final) > 0:
                    boxes_filt = torch.stack(boxes_final)
                else:
                    boxes_filt = np.array([])

                pil_image = pil_data[global_idx][0]
                image = np.array(pil_image)

                H, W = image.shape[:2]
                for i in range(boxes_filt.shape[0]):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                if save_image:
                    # draw output image
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    for box, label in zip(boxes_filt, pred_phrases):
                        show_box(box.numpy(), plt.gca(), label)

                    plt.axis("off")
                    plt.savefig(
                        os.path.join(output_dir, f"{global_idx}.png"),
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=0.0,
                    )

                if dataset_name == "places365_val" or dataset_name == "places365_train":
                    img_path = dataset.imgs[global_idx]
                else:
                    img_path = None

                save_mask_data(output_dir, f"{global_idx}.json", boxes_filt, pred_phrases, img_path)
