import json
import os
from typing import List, Optional, Tuple

import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import datasets, models, transforms
from tqdm import tqdm
from loguru import logger

import clip

# get from the environment variable
DATASET_FOLDER = os.environ.get("DATASET_FOLDER", "/datasets")

DATASET_ROOTS = {
    "imagenet_train": f"{DATASET_FOLDER}/CLS-LOC/train/",
    "imagenet_val": f"{DATASET_FOLDER}/ImageNet_val/",
    "cub_train": f"{DATASET_FOLDER}/CUB/train",
    "cub_val": f"{DATASET_FOLDER}/CUB/test",
}

LABEL_FILES = {
    "places365": "data/categories_places365_clean.txt",
    "imagenet": "data/imagenet_classes.txt",
    "cifar10": "data/cifar10_classes.txt",
    "cifar100": "data/cifar100_classes.txt",
    "cub": "data/cub_classes.txt",
}

PER_CLASS_CONCEPT_FILES = {
    "cifar10": "data/cifar10_per_class.json",
    "cifar100": "data/cifar100_per_class.json",
    "cub": "data/cub_per_class.json",
    "places365": "data/places365_per_class.json",
}

BACKBONE_ENCODING_DIMENSION = {
    "resnet18_cub": 512,
    "clip_RN50": 1024,
    "clip_RN50_penultimate": 2048,
    "resnet50": 2048,
}


def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=target_mean, std=target_std),
        ]
    )
    return preprocess


def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=True,
            transform=preprocess,
        )

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=False,
            transform=preprocess,
        )

    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=True,
            transform=preprocess,
        )

    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(
            root=os.path.expanduser(DATASET_FOLDER),
            download=True,
            train=False,
            transform=preprocess,
        )

    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="train-standard",
                small=True,
                download=True,
                transform=preprocess,
            )
        except RuntimeError:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="train-standard",
                small=True,
                download=False,
                transform=preprocess,
            )
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="val",
                small=True,
                download=True,
                transform=preprocess,
            )
        except RuntimeError:
            data = datasets.Places365(
                root=f"{os.path.expanduser(DATASET_FOLDER)}/places365_torch",
                split="val",
                small=True,
                download=False,
                transform=preprocess,
            )

    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)

    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset(
            [
                datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess),
                datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess),
            ]
        )
    return data


def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    return pil_data.targets


def get_target_model(target_name, device):
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()

    elif target_name == "resnet18_places":
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load("data/resnet18_places365.pth.tar")["state_dict"]
        new_state_dict = {}
        for key in state_dict:
            if key.startswith("module."):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif target_name == "resnet18_cub":
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    return target_model, preprocess


def format_concept(s):
    # replace - with ' '
    # replace , with ' '
    # only one space between words
    s = s.lower()
    s = s.replace("-", " ")
    s = s.replace(",", " ")
    s = s.replace(".", " ")
    s = s.replace("(", " ")
    s = s.replace(")", " ")
    if s[:2] == "a ":
        s = s[2:]
    elif s[:3] == "an ":
        s = s[3:]

    # remove trailing and leading spaces
    s = " ".join(s.split())
    return s


def get_per_class_concepts(dataset_name):
    with open(PER_CLASS_CONCEPT_FILES[dataset_name], "r") as f:
        per_class_concepts = json.load(f)
    for key, value in per_class_concepts.items():
        per_class_concepts[key] = [format_concept(concept) for concept in value]
        per_class_concepts[key] = list(set(per_class_concepts[key]))
    return per_class_concepts


def get_classes(dataset_name):
    with open(LABEL_FILES[dataset_name], "r") as f:
        classes = f.read().split("\n")
    return classes


def get_concepts(concept_file: str, filter_file:Optional[str]=None) -> List[str]:
    with open(concept_file) as f:
        concepts: List[str] = f.read().split("\n")

    # remove repeated concepts and maintain order
    concepts = list(dict.fromkeys([format_concept(concept) for concept in concepts]))

    # check for filter file
    if filter_file and os.path.exists(filter_file):
        logger.info(f"Filtering concepts using {filter_file}")
        with open(filter_file) as f:
            to_filter_concepts = f.read().split("\n")
        to_filter_concepts = [format_concept(concept) for concept in to_filter_concepts]
        concepts = [concept for concept in concepts if concept not in to_filter_concepts]

    return concepts


def save_concept_count(
    concepts: List[str],
    counts: List[int],
    save_dir: str,
    file_name: str = "concept_counts.txt",
):
    with open(os.path.join(save_dir, file_name), "w") as f:
        if len(concepts) != len(counts):
            raise ValueError("Length of concepts and counts should be the same")
        f.write(f"{concepts[0]} {counts[0]}")
        for concept, count in zip(concepts[1:], counts[1:]):
            f.write(f"\n{concept} {count}")


def load_concept_and_count(
    save_dir: str, file_name: str = "concept_counts.txt", filter_file:Optional[str]=None
) -> Tuple[List[str], List[float]]:
    with open(os.path.join(save_dir, file_name), "r") as f:
        lines = f.readlines()
        concepts = []
        counts = []
        for line in lines:
            concept = line.split(" ")[:-1]
            concept = " ".join(concept)
            count = line.split(" ")[-1]
            concepts.append(format_concept(concept))
            counts.append(float(count))

    if filter_file and os.path.exists(filter_file):
        with open(filter_file) as f:
            logger.info(f"Filtering concepts using {filter_file}")
            to_filter_concepts = f.read().split("\n")
        to_filter_concepts = [format_concept(concept) for concept in to_filter_concepts]
        counts = [count for concept, count in zip(concepts, counts) if concept not in to_filter_concepts]
        concepts = [concept for concept in concepts if concept not in to_filter_concepts]
        assert len(concepts) == len(counts)

    return concepts, counts

def save_filtered_concepts(
    filtered_concepts: List[str],
    save_dir: str,
    file_name: str = "filtered_concepts.txt",
):
    with open(os.path.join(save_dir, file_name), "w") as f:
        if len(filtered_concepts) > 0:
            f.write(filtered_concepts[0])
            for concept in filtered_concepts[1:]:
                f.write("\n" + concept)
