import logging
import os
import subprocess
from typing import TYPE_CHECKING, Iterable

from dotenv import load_dotenv

if TYPE_CHECKING:
    import numpy as np
    from faster_whisper.transcribe import Segment, TranscriptionInfo


# curl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o kokoro-v1.0.onnx
# curl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o voices-v1.0.bin
# install poetry, project dependencies and ffmpeg


def get_text_google(subject: str, api_key: str) -> str | None:
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
    return (
        response.text.replace("*", "")
        + " Don't forget to like and follow for more daily content!"
    )


def get_audio(text: str, voice: str) -> "np.ndarray":
    from kokoro_onnx import Kokoro

    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.15, lang="en-us")
    return samples, sample_rate


def play_audio(samples: "np.ndarray", sample_rate: int) -> None:
    import sounddevice as sd

    sd.play(samples, sample_rate)
    sd.wait()


def save_audio(samples: "np.ndarray", sample_rate: int, filename: str) -> None:
    from scipy.io import wavfile

    wavfile.write(filename, sample_rate, samples)


def generate_subtitles(filename: str) -> "tuple[Iterable[Segment], TranscriptionInfo]":
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(filename)
    return segments, info


def log_subtitles(segments: "Iterable[Segment]") -> None:
    for segment in segments:
        print(f"{segment.start:.3f} - {segment.end:.3f}: {segment.text}")


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


def download_video(url: str, resolution: str) -> None:
    from pytubefix import YouTube
    from pytubefix.cli import on_progress

    yt = YouTube(url, on_progress_callback=on_progress)

    for idx, i in enumerate(yt.streams):
        if i.resolution == resolution:
            break

    yt.streams[idx].download()


def process_video(source: str, length: int, output: str):
    import json
    import random

    duration_cmd = f'ffprobe -v quiet -print_format json -show_format "{source}"'
    duration_output = subprocess.check_output(duration_cmd, shell=True)
    duration_data = json.loads(duration_output)
    total_duration = float(duration_data["format"]["duration"])
    max_start = total_duration - length
    start_time = random.uniform(0, max_start)

    ffmpeg_cmd = (
        f'ffmpeg -ss {start_time:.2f} -i "{source}" -t {length} '
        f'-vf "crop=ih*9/16:ih,scale=1080:1920" -c:v h264_videotoolbox '
        f'-c:a aac -b:a 128k -preset faster "{output}" -y'
    )

    subprocess.run(ffmpeg_cmd, shell=True, check=True)


def add_audio(input_video: str, audio_file: str) -> None:
    temp_output = "temp_" + input_video  # Temporary file to hold the result

    command = [
        "ffmpeg",
        "-i",
        input_video,  # Input video file
        "-i",
        audio_file,  # Input audio file
        "-c:v",
        "copy",  # Copy video stream without re-encoding
        "-c:a",
        "aac",  # Encode audio to AAC format
        "-map",
        "0:v:0",  # Use video from the first input (video file)
        "-map",
        "1:a:0",  # Use audio from the second input (audio file)
        "-y",  # Overwrite the temporary output file
        temp_output,  # Temporary output file
    ]

    subprocess.run(command, check=True)

    # Rename the temporary file to the original file to overwrite it
    os.replace(temp_output, input_video)


def add_subtitles(input_video: str, subtitle_file: str) -> None:
    temp_output = "temp_" + input_video  # Temporary file to hold the result

    command = [
        "ffmpeg",
        "-i",
        input_video,  # Input video file
        "-vf",
        f"subtitles={subtitle_file}:force_style='Fontname=BebasNeue-Regular.ttf,PrimaryColour=&HFFFFFF&,BorderStyle=0,BorderWidth=0,Alignment=2,VerticalAlignment=1,MarginV=100'",  # Subtitles filter with size 90 and margin 100
        "-c:a",
        "copy",  # Copy audio stream without re-encoding
        "-y",  # Overwrite output file
        temp_output,  # Temporary output file
    ]
    subprocess.run(command, check=True)

    os.replace(temp_output, input_video)


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.CRITICAL)

    voice = "am_michael"
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    origin_video = "input.mp4"
    subject = "Give me 5 tips billionaires use to manage their time."
    title = "hello"

    srt_file = title + ".srt"
    audio_file = title + ".wav"
    video_file = title + ".mp4"

    # # Get the text from the subject
    # logging.critical("Getting text from Google")
    # text = get_text_google(subject, gemini_api_key)
    # logging.critical(f"Text: {text}")

    # # Get the audio from the text
    # logging.critical("Getting audio from the text")
    # samples, sample_rate = get_audio(text, voice)
    # save_audio(samples, sample_rate, audio_file)
    # logging.critical(f"Audio saved to {audio_file}")

    # # Generate subtitles
    # logging.critical("Getting subtitles from the audio")
    # segments, critical = generate_subtitles(audio_file)
    # save_subtitles(segments, srt_file)
    # logging.critical(f"Subtitles saved to {srt_file}")

    # # Download the video
    # logging.critical("Getting video from local file")
    # process_video(origin_video, len(samples) / sample_rate, video_file)
    # add_audio(video_file, audio_file)
    add_subtitles(video_file, srt_file)
    logging.critical(f"Video saved to {video_file}")
