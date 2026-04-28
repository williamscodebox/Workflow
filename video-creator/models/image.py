import os
import torch
import time
from diffusers import StableDiffusionPipeline
from typing import List


class ImageModel:
    def __init__(self, storage_path: str = ""):
        self.storage_path = storage_path

        # Auto-detect GPU backend
        if torch.cuda.is_available():
            self.device = "cuda"
            print("⚡ Using NVIDIA GPU (CUDA)")
        else:
            try:
                import torch_directml
                self.device = "dml"
                print("⚡ Using DirectML GPU (AMD/Intel)")
            except ImportError:
                self.device = "cpu"
                print("🐢 No GPU backend found — using CPU")

        print("🔄 Loading image model...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            safety_checker=None,
            feature_extractor=None,
        ).to(self.device)

        print("✅ Image Model loaded on", self.device)

        self.generator = torch.manual_seed(42)

        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    def generate_image(self, prompt: str, filename: str):
        outfile_path = f"{self.storage_path}{filename}.png"

        image = self.pipe(
            prompt=prompt,
            height=848,
            width=480,
            num_inference_steps=40,
            generator=self.generator,
        ).images[0]

        image.save(outfile_path)
        print(f"✅ Image saved as {outfile_path}")

    def generate_images(self, prompts: List[str], filename_prefix: str):
        for i, prompt in enumerate(prompts, start=1):
            self.generate_image(prompt, f"{filename_prefix}_{i}")
