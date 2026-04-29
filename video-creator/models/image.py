import os
import torch
import time
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL
from typing import List
from diffusers.utils import logging
logging.set_verbosity_error()  # <‑‑ disables ALL progress bars + logs


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

        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        )

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            vae=vae,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            safety_checker=None,
            feature_extractor=None,
        ).to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

        print("✅ Image Model loaded on", self.device)

        self.generator = torch.manual_seed(42)

        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    def generate_image(self, prompt: str, filename: str):
        outfile_path = f"{self.storage_path}{filename}.png"
        print("DEBUG — generating image for prompt:", repr(prompt))

        try:
            result = self.pipe(
                prompt=prompt,
                height=848,
                width=480,
                num_inference_steps=40,
                generator=self.generator,
            )
            image = result.images[0]
            image.save(outfile_path)
            print(f"✅ Image saved as {outfile_path}")
        except Exception as e:
            print("❌ IMAGE GENERATION FAILED:", type(e).__name__, "-", e)

    def generate_images(self, prompts: List[str], filename_prefix: str):
        print("DEBUG — generate_images called with", len(prompts), "prompts")
        for i, prompt in enumerate(prompts, start=1):
            self.generate_image(prompt, f"{filename_prefix}_{i}")
