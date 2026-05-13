from TTS.api import TTS
import soundfile as sf
import numpy as np
import os

class AudioModel:
    def __init__(self, storage_path="", model_name="tts_models/multilingual/multi-dataset/your_tts"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        print("[GlowTTS] Loading model…")

        # GlowTTS automatically loads HiFiGAN as vocoder
        self.tts = TTS(model_name).to("cuda")
        print("Available speakers:", self.tts.speakers)

        print("[GlowTTS] Model ready on GPU")

    def generate_audio(self, content: str, filename: str):
        outfile = f"{self.storage_path}{filename}.wav"
        print(f"[GlowTTS] Generating audio → {outfile}")

        # Generate audio (GlowTTS → HiFiGAN)
        audio = self.tts.tts(
            text=content,
            speaker="male-en-2",  # or any speaker ID
            language="en",
            length_scale=0.78,
            noise_scale=0.28,
            noise_w=0.60
        )

        # Ensure float32 numpy array
        audio = np.array(audio, dtype=np.float32)

        # Ensure shape is (samples,)
        if audio.ndim > 1:
            audio = audio.squeeze()

        # Save WAV
        sf.write(outfile, audio, self.tts.synthesizer.output_sample_rate)

        print(f"✅ Audio generated! {outfile}")
        return outfile
