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
                print("🐢 No GPU backend found — using CPU")

        print("🔄 Loading SDXL + ControlNet (OpenPose)...")

        # ---------- MODELS ----------
        # SDXL base
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        # SDXL OpenPose ControlNet
        controlnet_model_id = "diffusers/controlnet-openpose-sdxl-1.0"

        dtype = torch.float16 if self.device != "cpu" else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=dtype,
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
        )

        # Disable safety checker if you want full control
        self.pipe.safety_checker = None

        # Move to device
        if self.device == "cuda":
            self.pipe.to("cuda")
        elif self.device == "dml":
            # DirectML uses .to("dml") via torch_directml
            import torch_directml
            dml_device = torch_directml.device()
            self.pipe.to(dml_device)
        else:
            self.pipe.to("cpu")

        self.pipe.set_progress_bar_config(disable=True)

        # Seed for reproducibility
        self.generator = torch.manual_seed(42)

        # Offline flags (optional)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        print("✅ SDXL + ControlNet (OpenPose) loaded on", self.device)

    def generate_pose_image(
        self,
        prompt: str,
        pose_image_path: str,
        output_filename: str = "pose_instruction.png",
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        strength: float = 1.0,
    ) -> str:
        """
        Generate a movement-instruction image using SDXL + ControlNet (OpenPose).

        :param prompt: Text prompt describing the style and movement.
        :param pose_image_path: Path to an OpenPose skeleton / pose image (PNG/JPG).
        :param output_filename: Output file name (saved under storage_path).
        :return: Full path to the saved image.
        """

        if not os.path.exists(pose_image_path):
            raise FileNotFoundError(f"Pose image not found: {pose_image_path}")

        control_image = Image.open(pose_image_path).convert("RGB")

        # Core call
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
