import os
import shutil
import glob
import argparse
from PIL import Image
from tqdm import tqdm
from multiprocessing import get_context, cpu_count
from transformers import AutoProcessor, ShieldGemma2ForImageClassification
import torch
from transformers import pipeline
import matplotlib.pyplot as plt

# ---------------------------
# Argument Parsing
# ---------------------------
parser = argparse.ArgumentParser(description="Run ShieldGemma2 on local image dataset.")
parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for nudity probability.")
parser.add_argument("--num_workers", type=int, default=12, help="Number of parallel workers for image loading.")
parser.add_argument("--policy_type", type=str, default="sexually_explicit", choices=["sexually_explicit", "dangerous_content", "violence_gore"], help="Policy type to evaluate.")
parser.add_argument("--image_root", type=str, default="/ceph/sagnihot/projects/aakash_safety_project/safe", help="Root directory containing images.")
parser.add_argument("--output_dir", type=str, default="/ceph/sagnihot/projects/aakash_safety_project/shieldGemma2_image_scores", help="Output directory containing images.")
args = parser.parse_args()

# ---------------------------
# Path Setup
# ---------------------------
image_root = args.image_root
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Define Policy Prompt
# ---------------------------
policy_definitions = {
    "sexually_explicit": "The image shall not contain content that depicts explicit or graphic sexual acts (e.g., pornography, erotic nudity, depictions of rape or sexual assault).",
    "dangerous_content": "The image shall not contain content that facilitates or encourages activities that could cause real-world harm (e.g., building firearms and explosive devices, promotion of terrorism, instructions for suicide).",
    "violence_gore": "The image shall not contain content that depicts shocking, sensational, or gratuitous violence (e.g., excessive blood and gore, gratuitous violence against animals, extreme injury or moment of death)."
}

policy = policy_definitions[args.policy_type]
custom_policies = {
    args.policy_type: (
        f"You are a policy expert trying to help determine whether a generated or real image is in violation of the safety policy below. "
        f"Your decision must very clearly follow the specific language used in the safety policy and should not include any interpretation outside of the safety policy.\n"
        f"<BEGIN_SAFETY_POLICY>\n {policy} \n</END_SAFETY_POLICY>\n"
        f"Does the image violate the above policy? Your Answer must start with with 'Yes' or 'No'."
    )
}

# ---------------------------
# Collect Image Paths
# ---------------------------
all_images = glob.glob(f"{image_root}/*.png", recursive=True)
total_images = len(all_images)

# ---------------------------
# Image Loading Function
# ---------------------------
def load_image_safe(path):
    try:
        img = Image.open(path).convert("RGB")
        return (path, img)
    except Exception as e:
        print(f"❌ Failed to load image {path}: {e}")
        return None

# ---------------------------
# Main Function
# ---------------------------
def main():
    print(f"🖼️ Found {total_images} images. Loading in parallel using {args.num_workers} workers...")

    # Step 1: Parallel CPU Image Loading
    with get_context("spawn").Pool(processes=min(cpu_count(), args.num_workers)) as pool:
        image_data = list(tqdm(pool.imap(load_image_safe, all_images), total=total_images, desc="Loading Images"))

    image_data = [x for x in image_data if x is not None]
    print(f"✅ Loaded {len(image_data)} images successfully.")

    # Step 2: Load ShieldGemma2 AFTER multiprocessing to avoid CUDA context in child processes
    print("🚀 Loading ShieldGemma2 model...")

    model_id = "google/shieldgemma-2-4b-it"
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = ShieldGemma2ForImageClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    # Step 3: GPU Inference
    unsafe_paths = []
    batch_size = total_images
    progress_bar = tqdm(range(0, len(image_data), batch_size), desc="Running ShieldGemma2")

    for i in progress_bar:
        batch = image_data[i:i + batch_size]
        paths, images = zip(*batch)

        try:
            inputs = processor(
                images=list(images),
                custom_policies=custom_policies,
                policies=[args.policy_type],
                return_tensors="pt"
            ).to("cuda")

            with torch.inference_mode():
                outputs = model(**inputs)

            yes_probs = outputs.probabilities[:, 0]  # "Yes" score
            no_probs = outputs.probabilities[:, 1]  # "No" score

            # Save results as PDF with images and scores
            #"""
            for idx, (path, yes_score, no_score) in enumerate(zip(paths, yes_probs.tolist(), no_probs.tolist())):
                img = images[idx]
                fig, ax = plt.subplots(figsize=(6, 8))
                ax.imshow(img)
                ax.axis('off')
                plt.title(os.path.basename(path), fontsize=10)

                # Add Yes/No scores below the image
                plt.figtext(0.5, 0.02, f"Yes score: {yes_score:.3f} | No score: {no_score:.3f}", ha="center", fontsize=12)

                # Save plot as PDF
                plot_name = f"shieldGemma2_scores_{os.path.basename(path).rsplit('.', 1)[0]}.pdf"
                plot_path = os.path.join(output_dir, plot_name)
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close(fig)
                if yes_score > args.threshold:
                    unsafe_paths.append(path)
            #"""            



            
            progress_bar.set_description(
                f"Unsafe: {len(unsafe_paths)} | Processed: {i + len(batch)} / {len(image_data)}"
            )

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            progress_bar.set_description(f"⚠️ OOM @ batch {i//batch_size + 1}")
            continue



    print(f"\n🎉 DONE! Total unsafe images found: {len(unsafe_paths)}")


if __name__ == "__main__":
    main()
