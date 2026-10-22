"""Audio decoding and streaming SRT generation, also used by the GPU worker."""

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave


def timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def load_audio(video: Path):
    from faster_whisper.audio import decode_audio

    try:
        return decode_audio(str(video), sampling_rate=16000)
    except Exception as decode_error:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                f"Cannot decode audio ({decode_error}). Install FFmpeg: sudo dnf install ffmpeg-free"
            ) from decode_error
        print("Direct decoding failed; extracting the first audio track with FFmpeg.", file=sys.stderr)
        with tempfile.TemporaryDirectory(prefix="subtitle-audio-") as directory:
            audio = Path(directory) / "audio.wav"
            result = subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(video), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", str(audio)],
                capture_output=True, text=True, check=False,
            )
            if result.returncode:
                raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr.strip()}")
            # Read PCM directly: this also recovers from PyAV API incompatibilities.
            import numpy as np

            with wave.open(str(audio), "rb") as stream:
                return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(
                    np.float32
                ) / 32768.0


def write_transcription(model, audio, output: Path, language: str | None) -> int:
    from tqdm import tqdm

    segments, info = model.transcribe(
        audio, language=language, beam_size=5, vad_filter=True, word_timestamps=True,
    )
    label = "Language" if language else "Detected language"
    print(f"{label}: {info.language} (confidence {info.language_probability:.1%})", file=sys.stderr)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream, tqdm(
        total=info.duration, desc="Transcribing", unit="audio s", dynamic_ncols=True,
    ) as progress:
        for segment in segments:
            text = " ".join(segment.text.split())
            if text:
                if not math.isfinite(segment.start) or not math.isfinite(segment.end):
                    raise ValueError("Transcription returned a non-finite timestamp")
                start = max(0, round(segment.start * 1000))
                end = max(start + 1, round(segment.end * 1000))
                count += 1
                stream.write(f"{count}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n\n")
            position = min(info.duration, max(progress.n, segment.end))
            progress.update(position - progress.n)
        progress.update(max(0, info.duration - progress.n))
    return count


def create_model(model: str, device: str, offline: bool, cache_dir: str | None):
    from faster_whisper import WhisperModel

    print(f"Loading {model} on {device}...", file=sys.stderr)
    return WhisperModel(
        model, device=device, compute_type="float16" if device == "cuda" else "int8",
        local_files_only=offline, download_root=cache_dir,
    )


def main() -> int:
    """Run CUDA in a subprocess so native library crashes permit CPU recovery."""
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language")
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        audio = load_audio(args.video)
        model = create_model(args.model, "cuda", args.offline, args.cache_dir)
        count = write_transcription(model, audio, args.output, args.language)
        if not count:
            print("No speech detected; writing an empty SRT.", file=sys.stderr)
        return 0
    except Exception as error:
        print(f"CUDA attempt failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
