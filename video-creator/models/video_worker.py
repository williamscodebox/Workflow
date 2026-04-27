# video_worker.py
import sys
import json
from video import VideoModel   # your class

def main():
    args = json.loads(sys.argv[1])

    audio = args["audio"]
    images = args["images"]
    bgm = args.get("bgm")
    subtitles = args.get("subtitles")
    output = args["output"]

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
