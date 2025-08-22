# Safety Classification

Using ShieldGemma2-4b and Qwen2.5-7b-Instruct for classifying images as safe/unsafe.

## Installation
`conda env create -f environment.yml`

## Usage
Please use with a GPU.

1. `conda activate safe_gen`

### To use ShieldGemma2-4b
`python shieldGemma2_scores.py --policy_type "sexually_explicit" --image_root "/path/to/your/images" --output_dir "/path/to/your/output_dir"`

### To use Qwen2.5-7b-Instruct
`python qwen_scores.py --policy_type "sexually_explicit" --image_root "/path/to/your/images" --output_dir "/path/to/your/output_dir"`