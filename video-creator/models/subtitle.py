import subprocess
import json
import os
import tempfile

WHISPERX_PYTHON = r"C:\Users\panda\PycharmProjects\Workflow\.whisperXvenv\Scripts\python.exe"

class SubtitleModel:
    # noinspection PyMethodMayBeStatic
    def generate_subtitle(self, audio_path: str):
        """
        Calls WhisperX from the isolated venv and returns word-level subtitles.
        """

        # Temporary output file
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        #     output_json = tmp.name
        with tempfile.TemporaryDirectory() as tmpdir:

          # Run WhisperX
          cmd = [
              WHISPERX_PYTHON,
              "-m", "whisperx",
              audio_path,
              "--model", "large-v3",
              "--output_format", "json",
              "--output_dir", tmpdir,
              "--vad_method", "silero",
              "--no_align"
          ]

          # print("WHISPERX CMD:", cmd)

          subprocess.run(cmd, check=True)

          # WhisperX always names the file <audio>.json
          base = os.path.splitext(os.path.basename(audio_path))[0]
          json_path = os.path.join(tmpdir, f"{base}.json")

          # Load WhisperX output
          with open(json_path, "r", encoding="utf-8") as f:
              data = json.load(f)

        # Convert to your expected format
        words = []
        for segment in data.get("segments", []):
            for w in segment.get("words", []):
                words.append({
                    "word": w["text"],
                    "start": w["start"],
                    "end": w["end"],
                    "score": w.get("confidence", 1.0)
                })

        return words
