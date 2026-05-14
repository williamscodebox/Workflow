import os
import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
)
from PIL import Image


class PoseImageModel:
    def __init__(self, storage_path: str = "outputs/images"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        # ---------- DEVICE SELECTION ----------
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
                print("🐢 Using CPU")

        print("🔄 Loading SDXL + ControlNet (OpenPose)...")

        # ---------- MODELS ----------
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        controlnet_model_id = "thibaud/controlnet-openpose-sdxl-1.0"

        dtype = torch.float16 if self.device != "cpu" else torch.float32

        # Load ControlNet (correct way)
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=dtype,
        )

        # Load SDXL with ControlNet
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
        )

        # Disable safety checker
        self.pipe.safety_checker = None

        # Move to device
        if self.device == "cuda":
            self.pipe.to("cuda")
        elif self.device == "dml":
            import torch_directml
            dml_device = torch_directml.device()
            self.pipe.to(dml_device)
        else:
            self.pipe.to("cpu")

        self.pipe.set_progress_bar_config(disable=True)
        self.generator = torch.manual_seed(42)

        print("✅ SDXL + ControlNet (OpenPose) loaded on", self.device)

    def generate_pose_image(
        self,
        prompt: str,
        pose_image_path: str,
        output_filename: str = "pose_instruction.png",
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
    ) -> str:

        if not os.path.exists(pose_image_path):
            raise FileNotFoundError(f"Pose image not found: {pose_image_path}")

        control_image = Image.open(pose_image_path).convert("RGB")

        result = self.pipe(
            prompt=prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=self.generator,
        )

        image = result.images[0]
        output_path = os.path.join(self.storage_path, output_filename)
        image.save(output_path)

        print(f"🖼️ Pose instruction image saved to: {output_path}")
        return output_path


pose_model = PoseImageModel(storage_path="outputs/pose_images")

prompt = (
    "A clean minimal instructional fitness image of a person sitting on a couch "
    "tapping their toe. Plain white background, bright soft lighting, centered subject, "
    "clear leg and foot pose, high-contrast edges, no clutter, bold text 'TOE TAP', "
    "modern YouTube Shorts aesthetic."
)

pose_image_path = "poses/toe_tap_openpose.png"  # your OpenPose skeleton

pose_model.generate_pose_image(
    prompt=prompt,
    pose_image_path=pose_image_path,
    output_filename="toe_tap_instruction.png",
)


