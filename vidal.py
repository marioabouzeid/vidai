from typing import Iterable

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo
from kokoro_onnx import Kokoro
from scipy.io import wavfile

# curl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o kokoro-v1.0.onnx
# curl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o voices-v1.0.bin


def get_audio(text: str, voice: str) -> np.ndarray:
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    return samples, sample_rate


def play_audio(samples: np.ndarray, sample_rate: int) -> None:
    sd.play(samples, sample_rate)
    sd.wait()


def save_audio(samples: np.ndarray, sample_rate: int, filename: str) -> None:
    wavfile.write(filename, sample_rate, samples)


def generate_subtitles(filename: str) -> tuple[Iterable[Segment], TranscriptionInfo]:
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(filename)
    return segments, info


def log_subtitles(segments: Iterable[Segment]) -> None:
    for segment in segments:
        print(f"{segment.start:.3f} - {segment.end:.3f}: {segment.text}")


def save_subtitles(segments: Iterable[Segment], filename: str) -> None:
    srt_filename = filename.replace(".wav", ".srt")
    with open(srt_filename, "w") as srt_file:
        for i, segment in enumerate(segments):
            start = segment.start
            end = segment.end
            text = segment.text
            srt_file.write(f"{i + 1}\n")
            srt_file.write(
                f"{start // 3600:02}:{(start % 3600) // 60:02}:{start % 60:06.3f} --> {end // 3600:02}:{(end % 3600) // 60:02}:{end % 60:06.3f}\n"
            )
            srt_file.write(f"{text}\n\n")


if __name__ == "__main__":
    text = "Hello, world! I am getting started with VidA, a simple text to video tool. This tool is built using Python and ONNX. I hope you enjoy it! It is built for developers who want to create videos programmatically."
    voice = "am_michael"
    filename = "hello.wav"

    samples, sample_rate = get_audio(text, voice)
    play_audio(samples, sample_rate)
    save_audio(samples, sample_rate, filename)

    segments, info = generate_subtitles(filename)
    log_subtitles(segments)
    save_subtitles(segments, filename)
