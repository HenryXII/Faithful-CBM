<!-- omit in toc -->
# Building more faithful CBM with attribution map concept bounding box alignment and concept set refinement.

This is the repository for my Capstone project where I developed a novel automated faithfulness measurement combined with concept set refinement to further improve VLG-CBM.

**VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance_, NeurIPS 2024.** [[Project Website]](https://lilywenglab.github.io/VLG-CBM/) [[Paper]](https://arxiv.org/pdf/2408.01432)

<!-- omit in toc -->
## Setup

1. Setup conda environment and install dependencies

```bash
  conda create -n faithful-cbm python=3.12
  conda activate faithful-cbm
  pip install -r requirements.txt
```

2. Install Grounding DINO for generating annotations on custom datasets

```
git clone https://github.com/IDEA-Research/GroundingDINO
cd GroundingDINO
pip install -e .
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
cd ..
```

3. Download CUB dataset with `bash download_cub.sh`

## Running the models

1. Generate concepts annotated dataset with:
```
python -m scripts.generate_annotations --dataset cub_train --device cuda --batch_size 16 --text_threshold 0.25 --output_dir annotations

python -m scripts.generate_annotations --dataset cub_val --device cuda --batch_size 16 --text_threshold 0.25 --output_dir annotations
```
2. Train a concept bottleneck model using the config files in ./configs. For instance, to train a CUB model, run the following command:
```
python train_cbm.py --config configs/cub.json --annotation_dir annotations
```
3. Evaluate accuracy under different number Number of Effective Concepts (NEC):
```
python sparse_evaluation.py --load_path <path-to-model-dir> --lam <lambda-value>
```
4. Evaluate and visualize faithfulness with GroundingDINO and GradCAM++
Run the last section "Experiment" in the notebook `spatial_gradcam.ipynb`

