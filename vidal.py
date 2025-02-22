import logging
import os
import subprocess
from os.path import expanduser
from typing import TYPE_CHECKING, Iterable

from dotenv import load_dotenv

if TYPE_CHECKING:
    import numpy as np
    from faster_whisper.transcribe import Segment, TranscriptionInfo

# curl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o kokoro-v1.0.onnx
# curl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o voices-v1.0.bin
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
                Add some hooks to make it more engaging, like 'Don't miss #3'.
                Limit to 1.2 minute of talking time at most.
                Do not include the word count in your reply.
            """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=subject,
    )

    logging.critical("Text is generated using Gemini AI")

    return (
        response.text.replace("*", "")
        + " Don't forget to like and follow for more daily content!"
    )


def get_audio(text: str, voice: str, onnx: str, bin: str) -> "np.ndarray":
    logging.critical("Getting audio from the text")

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


def save_subtitles(segments: "Iterable[Segment]", filename: str) -> None:
    def format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

    with open(filename, "w") as srt_file:
        for i, segment in enumerate(segments):
            start = segment.start
            end = segment.end
            text = segment.text
            srt_file.write(f"{i}\n")
            srt_file.write(f"{format_time(start)} --> {format_time(end)}\n")
            srt_file.write(f"{text.upper()}\n\n")

    logging.critical(f"Subtitles saved to {filename}")


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
    total_duration = float(duration_data["format"]["duration"])
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
        "-shortest",
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
        "mp3",
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
        f"subtitles={subtitle_file}:force_style='BorderStyle=1,Outline=0,"
        + f"Fontsize=18,FontName={font_name},PrimaryColour=&H00FFFFFF,MarginV=135'",
        "-c:a",
        "copy",
        "-y",
        temp_output,
    ]

    subprocess.run(
        command, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )

    os.replace(temp_output, input_video)


def copy_icloud(video_file: str, folder: str) -> None:
    import shutil

    logging.critical("Copying video to iCloud")

    shutil.copy(
        video_file,
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

    text = get_text_google(subject, GEMINI_API_KEY)
    samples, sample_rate = get_audio(text, voice, ONNX, VOICES)
    save_audio(samples, sample_rate, voice_file)
    segments, critical = generate_subtitles(voice_file)
    save_subtitles(segments, srt_file)
    cut_video_snippet(ORIGIN_VIDEO, len(samples) / sample_rate, video_file)
    add_voice_to_video(video_file, voice_file)
    add_music_to_video(video_file, MUSIC_DIR)
    add_subtitles(video_file, srt_file, FONT_NAME)
    copy_icloud(video_file, "videos")

    logging.critical(f"Video saved to {video_file}")
