import subprocess
import sys
import json
import os

# Use the environment you confirmed is correct
SUBTITLE_PYTHON = r"D:\Projects\Workflow\.whisperXvenv\Scripts\python.exe"

class SubtitleModel:
    def generate_subtitle(self, audio_path: str):
        cmd = [
            SUBTITLE_PYTHON,
            os.path.abspath(__file__),
            audio_path
        ]

        print("USING PYTHON:", SUBTITLE_PYTHON)
        print("RUNNING SCRIPT:", os.path.abspath(__file__))

        env = os.environ.copy()
        env["PATH"] = (
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;" +
            env["PATH"]
        )

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",     # ← FIX 1
            errors="ignore",      # ← FIX 2
            env=env
        )

        print("SUBTITLE STDOUT:", result.stdout)
        print("SUBTITLE STDERR:", result.stderr)

        result.check_returncode()
        stdout_clean = result.stdout

        # Find first '[' and last ']'
        start = stdout_clean.find('[')
        end = stdout_clean.rfind(']')

        if start == -1 or end == -1:
            print("RAW STDOUT:", stdout_clean)
            raise ValueError("No JSON array found in subtitle output")

        json_text = stdout_clean[start:end + 1]

        return json.loads(json_text)


# ---------------------------------------------------------
# RUNNER MODE (this executes inside .whisperXvenv)
# ---------------------------------------------------------
if __name__ == "__main__":
    audio_path = sys.argv[1]

    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    # model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True
    )

    output = []

    for seg in segments:
        for w in seg.words:
            output.append({
                "word": w.word,
                "start": w.start,
                "end": w.end
            })

    print(json.dumps(output, ensure_ascii=False))


# Use the environment you confirmed is correct
# SUBTITLE_PYTHON = r"D:\Projects\Workflow\.whisperXvenv\Scripts\python.exe"
#
# class SubtitleModel:
#     def generate_subtitle(self, audio_path: str):
#         cmd = [
#             SUBTITLE_PYTHON,
#             os.path.abspath(__file__),
#             audio_path
#         ]
#
#         print("USING PYTHON:", SUBTITLE_PYTHON)
#         print("RUNNING SCRIPT:", os.path.abspath(__file__))
#
#         env = os.environ.copy()
#         env["PATH"] = (
#                 r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;" +
#                 env["PATH"]
#         )
#
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             env=env
#         )
#
#         print("SUBTITLE STDOUT:", result.stdout)
#         print("SUBTITLE STDERR:", result.stderr)
#
#         result.check_returncode()
#         return json.loads(result.stdout)
#
#         # result.check_returncode()
#
#         # return json.loads(result.stdout)
#
