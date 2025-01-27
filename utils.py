import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_model, load_image, predict, annotate

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import transforms
from loguru import logger
from PIL import Image
import cv2

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import box_convert
import utils
import data.utils as data_utils

HOME="/home/jix049/private/"

def compute_saliency_map(model, x, target_neuron, requires_grad=True, mode="GradCAM++"):
    """
    Compute saliency map for a specific neuron in proj_layer for a single input image.
    
    Args:
        model (CBM_model): The CBM model
        x (torch.Tensor): Single input image tensor of shape [channels, height, width]
        target_neuron (int): Index of the neuron in proj_layer to visualize
        requires_grad (bool): Whether to compute gradients for the input
    
    Returns:
        numpy.ndarray: Saliency map for the specified neuron
    """
    if mode == "GradCAM++":
        target_layer = [model[0].backbone.features.stage4.unit2.body.conv2.conv] # hardcoded for CUB-VLG-CBM
        cam = GradCAMPlusPlus(
            model=model,
            target_layers=target_layer,
        )
        targets = [ClassifierOutputTarget(target_neuron)]
        # For a single image
        if x.dim() == 3:
            x_4d = x.unsqueeze(0)
        # Make sure image is on the same device as the model
        grayscale_cam = cam(input_tensor=x_4d, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam
    if mode == "gradient":
        # Ensure the model is in eval mode
        model.eval()
        
        # Create a copy of the input that requires gradient
        if requires_grad:
            x = x.clone().detach().requires_grad_(True)
        
        # Add batch dimension
        x_batch = x.unsqueeze(0)  # Shape becomes [1, channels, height, width]
        
        # Forward pass through backbone and proj_layer
        features = model.backbone(x_batch)
        # flatten not needed for VLG_CBM, but is needed for LF_CBM
        features_flat = torch.flatten(features, 1)
        proj_output = model.proj_layer(features_flat)
        
        # Get the output for the specified neuron
        neuron_output = proj_output[0, target_neuron]  # Take first (only) element since we have batch size 1
        
        # Compute gradients
        neuron_output.backward()
        
        # Get gradients with respect to input
        gradients = x.grad.detach()
        
        # Compute saliency map (take absolute value and max across channels)
        saliency_map = torch.max(torch.abs(gradients), dim=0)[0]
        
        # Convert to numpy and normalize to [0, 1]
        saliency_map = saliency_map.cpu().numpy()
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map

def visualize_saliency_map(model, x, target_neuron, normalize_input=True):
    """
    Visualize the saliency map alongside the original image.
    
    Args:
        model (CBM_model): The CBM model
        x (torch.Tensor): Single input image tensor of shape [channels, height, width]
        target_neuron (int): Index of the neuron in proj_layer to visualize
        normalize_input (bool): Whether to normalize the input image for visualization
    """
    # Compute saliency map
    saliency_map = compute_saliency_map(model, x, target_neuron)
    
    # Convert input image to numpy and move to CPU
    img = x.detach().cpu().numpy()
    
    # Move channels to last dimension and normalize if needed
    img = np.transpose(img, (1, 2, 0))
    if normalize_input:
        img = (img - img.min()) / (img.max() - img.min())
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original image
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot saliency map
    saliency_plot = ax2.imshow(saliency_map, cmap='hot')
    ax2.axis('off')
    
    # Add colorbar
    plt.colorbar(saliency_plot, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def load_annotation_model(model_config_path=HOME+"sandbox-VLG-CBM/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", model_checkpoint_path=HOME+"sandbox-VLG-CBM/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth", device="cuda"):
    """
    return groundingDino and tokenizer
    """
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    logger.info(load_res)
    model.eval()
    model.to(device)
    return model

def transform_boxes(boxes, original_size, target_size=(224, 224)):
    """
    Transform boxes from Grounding DINO preprocessing to ResNet preprocessing
    
    Args:
        boxes: tensor of shape (N, 4) in [center_x, center_y, width, height] format, normalized to [0,1]
        original_size: tuple of (height, width) after Grounding DINO's 800px resize
        target_size: tuple of (height, width) for final ResNet size (224, 224)
    """
    if boxes is None:
        return None
    # Convert from cxcywh to xyxy
    boxes_xyxy = box_convert(boxes, 'cxcywh', 'xyxy')
    
    # 1. Denormalize boxes to get pixel coordinates in 800px-resized space
    boxes_xyxy = boxes_xyxy * torch.tensor([original_size[1], original_size[0], 
                                          original_size[1], original_size[0]])
    
    # 2. Scale factor from 800px to 256px resize
    scale = 256 / min(original_size)
    new_h, new_w = int(original_size[0] * scale), int(original_size[1] * scale)
    boxes_xyxy = boxes_xyxy * scale
    
    # 3. Adjust for center crop from (new_h, new_w) to (224, 224)
    x_offset = (new_w - 224) // 2
    y_offset = (new_h - 224) // 2
    
    boxes_xyxy[:, [0, 2]] -= x_offset  # adjust x coordinates
    boxes_xyxy[:, [1, 3]] -= y_offset  # adjust y coordinates
    
    # 4. Clip to image boundaries
    boxes_xyxy = torch.clamp(boxes_xyxy, min=0, max=224)
    
    # 5. Normalize to [0,1] for final 224x224 image
    boxes_xyxy = boxes_xyxy / 224
    
    # Convert back to cxcywh
    boxes_cxcywh = box_convert(boxes_xyxy, 'xyxy', 'cxcywh')
    
    return boxes_cxcywh

def compute_saliency_box_alignment(saliency_map, boxes, image_size):
    """
    Compute metrics for how well a saliency map aligns with multiple bounding boxes.
    Uses the union of all boxes for computation.
    
    Args:
        saliency_map (numpy.ndarray): 2D array of saliency values, normalized to [0,1]
        boxes (torch.Tensor): Multiple bounding boxes in format [N, 4] where each box is
                            [center_x, center_y, width, height] normalized to [0,1]
        image_size (tuple): Tuple of (height, width) of the image/saliency map
    
    Returns:
        dict: Dictionary containing various alignment metrics
    """
    height, width = image_size
    
    # Create binary mask for the union of all boxes
    box_mask = np.zeros_like(saliency_map)
    
    # Process each box and update the union mask
    for box in boxes:
        # Convert center coordinates to pixels
        center_x = int(box[0] * width)
        center_y = int(box[1] * height)
        box_width = int(box[2] * width)
        box_height = int(box[3] * height)
        
        # Calculate box boundaries
        x1 = max(0, int(center_x - box_width/2))
        y1 = max(0, int(center_y - box_height/2))
        x2 = min(width, int(center_x + box_width/2))
        y2 = min(height, int(center_y + box_height/2))
        
        # Update union mask
        box_mask[y1:y2, x1:x2] = 1
    
    # Calculate metrics using the union mask
    
    # 1. IoU (Intersection over Union) with thresholded saliency map
    threshold = np.percentile(saliency_map, 70)  # Use top 30% of saliency values
    binary_saliency = (saliency_map > threshold).astype(float)
    
    intersection = np.sum(binary_saliency * box_mask)
    union = np.sum((binary_saliency + box_mask) > 0)
    iou = intersection / (union + 1e-8)
    
    # 2. Average saliency within boxes vs outside boxes
    in_box_mask = box_mask > 0
    outside_box_mask = box_mask == 0
    
    # Handle cases where box might cover entire image or be completely outside
    if np.any(in_box_mask):
        saliency_in_box = np.mean(saliency_map[in_box_mask])
    else:
        saliency_in_box = 0
        
    if np.any(outside_box_mask):
        saliency_outside_box = np.mean(saliency_map[outside_box_mask])
    else:
        saliency_outside_box = 0
    
    # Avoid division by zero
    if saliency_outside_box > 0:
        saliency_ratio = saliency_in_box / saliency_outside_box
    else:
        saliency_ratio = saliency_in_box if saliency_in_box > 0 else 0
    
    # 3. Percentage of total saliency captured by boxes
    total_saliency = np.sum(saliency_map)
    saliency_in_box_total = np.sum(saliency_map * box_mask)
    
    # Avoid division by zero
    if total_saliency > 0:
        saliency_capture = saliency_in_box_total / total_saliency
    else:
        saliency_capture = 0
    
    return {
        'iou': float(iou),
        'saliency_ratio': float(saliency_ratio),
        'saliency_capture': float(saliency_capture),
    }

def evaluate_concept_saliency_alignment(model, image, boxes, target_neuron):
    """
    Evaluate saliency map alignment with concept bounding boxes for multiple concepts.
    
    Args:
        model: The model to generate saliency maps
        image (torch.Tensor): Input image tensor [C, H, W]
        concept (str): The concept to evaluate
        boxes (torch.Tensor): Bounding boxes tensor [N, 4]
        phrases (list): List of concept phrases
        target_neuron (int): Neuron index corresponding to the concept
    
    Returns:
        tuple: (Dictionary containing alignment metrics, saliency map)
    """
    image_size = (image.shape[1], image.shape[2])  # (H, W)
    
    if boxes is None:
        return {
            'iou': 0,
            'saliency_ratio': 0,
            'saliency_capture': 0,
        }, None
    
    # Compute saliency map for this concept's neuron
    saliency_map = compute_saliency_map(model, image, target_neuron)
    
    # Compute alignment using all matching boxes
    result = compute_saliency_box_alignment(saliency_map, boxes, image_size)
    
    return result, saliency_map

def get_matching_boxes_and_phrases(boxes, phrases, concept):
    """
    Filter boxes and phrases to only those matching the given concept.
    
    Args:
        boxes (torch.Tensor): Bounding boxes tensor [N, 4] where each box is
                            [center_x, center_y, width, height]
        phrases (list): List of concept phrases
        concept (str): The concept to match
        
    Returns:
        tuple: (matching_boxes, matching_phrases)
            - matching_boxes (torch.Tensor): Tensor of boxes that match the concept [M, 4]
            - matching_phrases (list): List of matched phrases
    """
    matching_boxes = []
    matching_phrases = []
    
    for box, phrase in zip(boxes, phrases):
        # if phrase == concept:
        if concept in phrase: # sometimes phrase could be combined concepts (bird red head)
            matching_boxes.append(box)
            matching_phrases.append(phrase)
    
    # If we found matches, stack the boxes into a tensor
    if matching_boxes:
        matching_boxes = torch.stack(matching_boxes) if len(matching_boxes) > 1 else matching_boxes[0].unsqueeze(0)
    else:
        matching_boxes = None
        
    return matching_boxes, matching_phrases

import matplotlib.patches as patches
def visualize_saliency_box_alignment(image, saliency_map, boxes, concept):
    """
    Visualize the image with both bounding boxes and saliency overlay (left)
    and saliency map (right), maximizing image size.
    
    Args:
        image (torch.Tensor): Input image tensor [C, H, W]
        saliency_map (numpy.ndarray): 2D array of saliency values
        boxes (torch.Tensor): Bounding boxes in format [N, 4] where each box is
                            [center_x, center_y, width, height]
        concept_name (str): Name of the concept for the title
    """
    # Convert image to numpy and move channels to last dimension
    img = image.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    
    # Normalize image for visualization
    img = (img - img.min()) / (img.max() - img.min())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Original image with both bounding boxes and saliency overlay
    ax1.imshow(img)
    
    # Generate distinct colors for multiple boxes
    num_boxes = len(boxes) if isinstance(boxes, (list, torch.Tensor)) else 1
    if num_boxes == 1 and not isinstance(boxes, (list, torch.Tensor)):
        boxes = [boxes]
    
    # Create color gradient from light to dark green
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, num_boxes))
    
    height, width = img.shape[:2]
    # Draw each box with a different shade of green
    for idx, box in enumerate(boxes):
        # Convert box coordinates to pixels
        center_x = box[0] * width
        center_y = box[1] * height
        box_width = box[2] * width
        box_height = box[3] * height
        
        # Calculate rectangle coordinates
        rect_x = center_x - box_width/2
        rect_y = center_y - box_height/2
        
        # Add rectangle patch with color from gradient
        rect = patches.Rectangle(
            (rect_x, rect_y),
            box_width,
            box_height,
            linewidth=2,
            edgecolor=colors[idx],
            facecolor='none'
        )
        ax1.add_patch(rect)
    
    # Add saliency overlay
    threshold = np.percentile(saliency_map, 70)
    binary_saliency = (saliency_map > threshold).astype(float)
    mask = np.zeros((*img.shape[:2], 4))  # RGBA
    mask[binary_saliency > 0] = [1, 0, 0, 0.3]  # Semi-transparent red
    ax1.imshow(mask)

    ax1.set_title(f'Concept: "{concept}"')
    ax1.axis('off')
    
    # 2. Saliency map
    saliency_img = ax2.imshow(saliency_map, cmap='hot')
    ax2.set_title('Saliency')
    ax2.axis('off')
    
    # Add colorbar for saliency map
    plt.colorbar(saliency_img, ax=ax2, fraction=0.046, pad=0.04)
    
    # Adjust layout to minimize empty space
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.05)
    plt.tight_layout()
    plt.show()

def show_neurons(model, grounding_model, concepts, target_neurons, activation_save_path, dataset, dataset_pil, box_t=0.35, text_t=0.25, num_show=10, batch_size=128, target_layer='proj_layer', device='cuda'):
    """
    show spatial alignment for target_neurons in a model
    
    Args:
        model
        concepts (list) : list of concepts for cbl neurons
        target_neurons (list) : neurons you want to examinate
        save_path
    """
    all_activations = utils.save_summary_activations(model, dataset, device, target_layer, batch_size, activation_save_path, pool_mode="avg")
    ratio=[]
    for target_neuron in target_neurons:
        summary_activations = all_activations[:, target_neuron]
        sorted_act_vals, sorted_act_ids = torch.sort(summary_activations, descending=True)
        to_show = sorted_act_ids[:num_show]
        
        # Evaluate alignment
        for image_id in to_show:
            print(f"image:{image_id}")
            concept=data_utils.format_concept(concepts[target_neuron], mode="remove")
            TEXT_PROMPT = f"bird . {concept}" # possible improvement: use class names
            BOX_TRESHOLD = box_t
            TEXT_TRESHOLD = text_t
            
            # run groundingDino
            transform = T.Compose(
                [
                    # Resize(800),
                    T.RandomResize([800]),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            
            image_transformed, _=transform(dataset_pil[image_id][0],None)
            # image_source=np.array(dataset_pil[image_id][0].convert("RGB"))
            boxes, logits, phrases = predict(
                model=grounding_model,
                image=image_transformed,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                remove_combined=True
            )
            # only keep matching boxes
            boxes, phrases=get_matching_boxes_and_phrases(boxes, phrases, concept)
        
            image=dataset[image_id][0].cuda()
            boxes_transformed = transform_boxes(boxes, (image_transformed.shape[1],image_transformed.shape[2]))
            
            result, saliency_map = evaluate_concept_saliency_alignment(model, image, boxes_transformed, target_neuron)
            if saliency_map is not None:
                visualize_saliency_box_alignment(image, saliency_map, boxes_transformed, concept)
            else:
                display(f"{concept} not found!")
                display(dataset_pil[image_id][0])
            ratio.append(result["saliency_ratio"])
            formatted_result = {k: f"{v:.4f}" for k, v in result.items()}
            display(formatted_result)
        return ratio