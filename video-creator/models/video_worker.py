# video_worker.py
import sys
import json
from video import VideoModel   # your class
import os

def main():
    args = json.loads(sys.argv[1])

    audio = args["audio"]
    images = args["images"]
    bgm = args.get("bgm")
    subtitles = args.get("subtitles")
    output = args["output"]

    print("DEBUG — images folder:", images)
    print("DEBUG — files:", os.listdir(images))

    model = VideoModel(audio)
    model.set_video(images)

    if bgm:
        model.attach_audio(bgm)
    else:
        model.attach_audio()

    if subtitles:
        model.attach_subtitle(subtitles)

    model.generate_video(output)

if __name__ == "__main__":
    main()
