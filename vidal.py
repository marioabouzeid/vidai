import logging
import os
import subprocess
from os.path import expanduser
from typing import TYPE_CHECKING, Iterable

from dotenv import load_dotenv

if TYPE_CHECKING:
    import numpy as np
    from faster_whisper.transcribe import Segment, TranscriptionInfo

# curl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o lib/kokoro-v1.0.onnx
# curl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o lib/voices-v1.0.bin
# install poetry, project dependencies and ffmpeg

load_dotenv()
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(message)s [in %(filename)s:%(lineno)d]",
    datefmt="%Y-%m-%d %H:%M:%S",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ONNX, VOICES = "lib/kokoro-v1.0.onnx", "lib/voices-v1.0.bin"
FONT_NAME = "Bebas Neue"
ORIGIN_VIDEO = "lib/input.mp4"
MUSIC_DIR = "lib/audio"


def get_text_google(subject: str, api_key: str) -> str | None:
    logging.critical("Getting text from Google")

    from google import genai

    client = genai.Client(api_key=api_key)
    subject += """
                A man will be reading your answer directly so don't write 
                as if you are replying to me, just the answer directly.
                Add a supe enganging hook at the beginning, make up something random.
                Limit to 30 seconds of talking time at most (around 80 words).
                Do not include the word count in your reply.
            """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=subject,
    )

    words = response.text.split()
    logging.critical(f"Text is generated using Gemini AI, {len(words)} words")

    return (
        response.text.replace("*", "")
        + " Don't forget to like and follow for more daily content!"
    )


def get_audio(text: str, voice: str, onnx: str, bin: str) -> "np.ndarray":
    logging.critical("Generating voiceover from the text using Kokoro ONNX")

    from kokoro_onnx import Kokoro

    kokoro = Kokoro(onnx, bin)
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.15, lang="en-us")
    return samples, sample_rate


def play_audio(samples: "np.ndarray", sample_rate: int) -> None:
    import sounddevice as sd

    sd.play(samples, sample_rate)
    sd.wait()


def save_audio(samples: "np.ndarray", sample_rate: int, filename: str) -> None:
    from scipy.io import wavfile

    wavfile.write(filename, sample_rate, samples)
    logging.critical(f"Audio saved to {filename}")


def generate_subtitles(filename: str) -> "tuple[Iterable[Segment], TranscriptionInfo]":
    from faster_whisper import WhisperModel

    logging.critical("Getting subtitles from the audio")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(filename)
    return segments, info


def save_subtitles(segments: "Iterable[Segment]", sub_filename: str) -> None:
    def format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

    with open(sub_filename, "w") as srt_file:
        for i, segment in enumerate(segments):
            start = segment.start
            end = segment.end
            text = segment.text
            srt_file.write(f"{i}\n")
            srt_file.write(f"{format_time(start)} --> {format_time(end)}\n")
            srt_file.write(f"{text.upper()}\n\n")

    logging.critical(f"Subtitles saved to {sub_filename}")


def download_video(url: str, resolution: str) -> None:
    from pytubefix import YouTube
    from pytubefix.cli import on_progress

    yt = YouTube(url, on_progress_callback=on_progress)

    for idx, i in enumerate(yt.streams):
        if i.resolution == resolution:
            break

    yt.streams[idx].download()


def cut_video_snippet(source: str, length: int, output: str):
    import json
    import random

    logging.critical("Getting video from local file")
    duration_cmd = f'ffprobe -v quiet -print_format json -show_format "{source}"'
    duration_output = subprocess.check_output(duration_cmd, shell=True)
    duration_data = json.loads(duration_output)
    total_duration = float(duration_data["format"]["duration"]) + 3
    max_start = total_duration - length
    start_time = random.uniform(0, max_start)

    command = (
        f'ffmpeg -ss {start_time:.2f} -i "{source}" -t {length} '
        f'-vf "crop=ih*9/16:ih,scale=1080:1920" -c:v h264_videotoolbox '
        f'-c:a aac -b:a 128k -preset faster "{output}" -y'
    )

    subprocess.run(
        command,
        check=True,
        shell=True,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )


def add_voice_to_video(input_video: str, voice_file: str) -> None:
    logging.critical("Adding voiceover to the video")
    temp_output = input_video.replace(".mp4", "_temp.mp4")
    command = [
        "ffmpeg",
        "-i",
        input_video,
        "-i",
        voice_file,
        "-c",
        "copy",
        "-map",
        "0:v",
        "-map",
        "1:a",
        temp_output,
    ]

    subprocess.run(
        command, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    os.replace(temp_output, input_video)


def add_music_to_video(input_video: str, music_dir: str) -> None:
    import random

    logging.critical("Adding music to the video")
    music_file = os.path.join(music_dir, random.choice(os.listdir(music_dir)))
    temp_output = input_video.replace(".mp4", "_temp.mp4")

    command = [
        "ffmpeg",
        "-i",
        input_video,
        "-i",
        music_file,
        "-filter_complex",
        "[0:a]volume=1.5[a0]; [1:a]volume=0.05[a1]; [a0][a1]amerge=inputs=2[a]",
        "-map",
        "0:v",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        temp_output,
    ]
    subprocess.run(
        command, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    os.replace(temp_output, input_video)


def add_subtitles(input_video: str, subtitle_file: str, font_name: str) -> None:
    logging.critical("Adding subtitles to the video")
    temp_output = input_video.replace(".mp4", "_temp.mp4")

    command = [
        "ffmpeg",
        "-i",
        input_video,
        "-vf",
        f"subtitles={subtitle_file}:force_style='BorderStyle=1,Outline=0,Shadow=1,"
        + f"Fontsize=18,FontName={font_name},PrimaryColour=&H00FFFFFF,MarginV=125'",
        "-c:a",
        "copy",
        "-y",
        temp_output,
    ]

    subprocess.run(
        command, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )

    os.replace(temp_output, input_video)


def move_to_icloud(video_file: str, new_filename: str, folder: str) -> None:
    import shutil

    logging.critical("Copying video to iCloud")
    new_filename = video_file.split("/")[0:-1] + [f"{new_filename}.mp4"]
    new_filename = "/".join(new_filename)
    os.replace(video_file, new_filename)

    shutil.move(
        new_filename,
        expanduser("~") + f"/Library/Mobile Documents/com~apple~CloudDocs/{folder}",
    )

    logging.critical("Video copied to iCloud")


if __name__ == "__main__":
    subject = "Tips for a successful and wealthy life"
    voice = "am_michael"

    logging.critical(f"Starting VIDAI for: {subject}")
    srt_file = "out/input.srt"
    voice_file = "out/input.wav"
    video_file = "out/input.mp4"
    filename = subject.replace(" ", "_").lower()

    text = get_text_google(subject, GEMINI_API_KEY)
    samples, sample_rate = get_audio(text, voice, ONNX, VOICES)
    save_audio(samples, sample_rate, voice_file)
    segments, info = generate_subtitles(voice_file)
    save_subtitles(segments, srt_file)
    cut_video_snippet(ORIGIN_VIDEO, len(samples) / sample_rate, video_file)
    add_voice_to_video(video_file, voice_file)
    add_music_to_video(video_file, MUSIC_DIR)
    add_subtitles(video_file, srt_file, FONT_NAME)
    move_to_icloud(video_file, filename, "videos")

    logging.critical(f"Video saved to {video_file}")
