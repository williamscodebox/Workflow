import os
import torch
from diffusers import AutoPipelineForText2Image
from typing import List


class ImageModel:
    def __init__(self, storage_path: str = ""):
        self.storage_path = storage_path

        # Auto-detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
            print("⚡ Using NVIDIA GPU (CUDA)")
        else:
            try:
                import torch_directml  # type: ignore
                self.device = "dml"
                dtype = torch.float16
                print("⚡ Using DirectML GPU (AMD/Intel)")
            except ImportError:
                self.device = "cpu"
                dtype = torch.float32
                print("🐢 No GPU backend found — using CPU")

        print("🔄 Loading SDXL-Turbo image model...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=dtype,
        ).to(self.device)

        # Speed / memory tweaks
        if self.device != "cpu":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

        print(f"✅ SDXL-Turbo loaded on {self.device}")

        self.generator = torch.manual_seed(42)

        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    def generate_image(self, prompt: str, filename: str):
        outfile_path = f"{self.storage_path}{filename}.png"

        image = self.pipe(
            prompt=prompt,
            height=848,
            width=480,
            num_inference_steps=4,   # SDXL-Turbo is designed for very low steps
            guidance_scale=0.0,      # recommended for turbo
            generator=self.generator,
        ).images[0]

        image.save(outfile_path)
        print(f"✅ Image saved as {outfile_path}")

    def generate_images(self, prompts: List[str], filename_prefix: str):
        for i, prompt in enumerate(prompts, start=1):
            self.generate_image(prompt, f"{filename_prefix}_{i}")
