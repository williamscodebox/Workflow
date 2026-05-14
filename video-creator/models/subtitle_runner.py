import sys
import json
from faster_whisper import WhisperModel
from stable_whisper import align

audio_path = sys.argv[1]

# 1. Fast transcription
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe(audio_path, word_timestamps=False)

whisper_segments = [
    {"start": s.start, "end": s.end, "text": s.text}
    for s in segments
]

# 2. Word-level alignment
aligned = align(audio_path, whisper_segments, model_name="large-v3")

print(json.dumps(aligned, ensure_ascii=False))
