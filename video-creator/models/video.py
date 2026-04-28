from moviepy import *
import os
from typing import List

class VideoModel:
    def __init__(self, audio_path: str):
        self.audio = AudioFileClip(audio_path)
        self.audio_duration = self.audio.duration

    def set_video(self, image_folder: str):
        image_files = sorted(
            os.path.join(image_folder, f) for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if not image_files:
            raise RuntimeError(f"No images found in folder: {image_folder}")

        image_duration = self.audio_duration / len(image_files)
        image_clips = [
            VideoEffects.add_fade_effect(path, image_duration) for path in image_files
        ]
        base_video = concatenate_videoclips(image_clips, method="compose", padding=-0.5)

        # Calculate leftover time
        leftover = max(0, self.audio_duration - base_video.duration)

        filler_clip = (
            ImageClip("images/filler.png")
            .with_duration(leftover)
            .resized(base_video.size)
            .with_effects([
                vfx.FadeIn(0.2),
                vfx.FadeOut(0.2),
            ])
        )

        self.video = concatenate_videoclips([base_video, filler_clip], method="compose")

    def attach_audio(self, bgm_path: str | None = None):
        audios = [self.audio]

        if bgm_path:
            if not isinstance(bgm_path, str):
                raise TypeError(f"attach_audio expected a file path, got {type(bgm_path)}")

            print("DEBUG — bgm_path:", bgm_path)
            print("DEBUG — exists:", os.path.exists(bgm_path))

            bgm = (
                AudioFileClip(bgm_path)
                .max_volume(0.2)
                .set_duration(self.audio_duration)
            )
            audios.append(bgm)

        composite_audio = CompositeAudioClip(audios)
        self.video = self.video.set_audio(composite_audio)

    def attach_subtitle(self, subtitles: List):
        group_size = 4
        grouped = []

        for i in range(0, len(subtitles), group_size):
            group = subtitles[i : i + group_size]
            full_text = " ".join([w["word"] for w in group])
            grouped.append(
                {
                    "word": full_text,
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "words": group,
                }
            )

        self.video = SubtitleEffects.simple_subtitle(self.video, grouped)

    def generate_video(self, output_path: str):
        self.video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            audio=True,
        )

        print(f"🔊 Has audio: {self.video.audio is not None}")


class VideoEffects:

    @staticmethod
    def add_fade_effect(image_path, duration=3.0, fade_duration=0.5):
        clip = ImageClip(image_path).with_duration(duration)

        # MoviePy 2.0: with_effects expects *effect instances*, not tuples
        clip = clip.with_effects([
            vfx.FadeIn(fade_duration),
            vfx.FadeOut(fade_duration),
        ])

        return clip

    @staticmethod
    def apply_shaky_zoom(
        image_path, duration, zoom_start=1.0, zoom_end=1.1, fade_duration=0.5
    ):
        # Load image as clip
        clip = ImageClip(image_path).with_duration(duration).set_fps(24)

        # Apply zoom using a lambda-based resize (linear interpolation over time)
        zoomed = clip.resized(
            lambda t: zoom_start + (zoom_end - zoom_start) * (t / duration) ** 1.5
        )

        # Apply fade-in and fade-out
        zoomed = zoomed.fadein(fade_duration).fadeout(fade_duration)

        return zoomed

    @staticmethod
    def apply_stable_zoom(
        image_path, duration, zoom_start=1.0, zoom_end=1.1, fade_duration=0.5
    ):
        clip = ImageClip(image_path).resized(height=720).set_duration(duration)

        w, h = clip.size

        def crop_zoom(get_frame, t):
            factor = zoom_start + (zoom_end - zoom_start) * (t / duration)
            new_w, new_h = int(w / factor), int(h / factor)
            x_center, y_center = w // 2, h // 2
            return (
                clip.crop(
                    x_center=x_center, y_center=y_center, width=new_w, height=new_h
                )
                .resized((w, h))
                .get_frame(t)
            )

        zoomed = clip.fl(crop_zoom, apply_to=["mask"])

        return zoomed.fadein(fade_duration).fadeout(fade_duration)


class SubtitleEffects:
    @staticmethod
    def simple_subtitle(video, subtitle: List):
        subtitle_clips = []
        for word_info in subtitle:
            word = word_info["word"]
            start = word_info["start"]
            duration = word_info["end"] - word_info["start"]

            txt_clip = (
                TextClip(
                    word,
                    fontsize=50,
                    color="white",
                    bg_color="rgba(0,0,0,0.4)",
                    font="Impact",
                    stroke_color="black",
                    stroke_width=2,
                    method="caption",  # enables word wrapping
                    size=(
                        video.w - 100,
                        None,
                    ),  # wrap at screen width - 100px padding
                )
                .set_position(("center", "bottom"))
                .margin(bottom=80)
                .set_start(start)
                .set_duration(duration)
                .fadein(0.01)
            )

            subtitle_clips.append(txt_clip)

        return CompositeVideoClip([video] + subtitle_clips)
