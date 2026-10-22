import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from . import engine

VIDEO_EXTENSIONS = {".mkv", ".mp4"}


def discover_videos(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Input file must be MKV or MP4")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input path does not exist or is not a folder: {path}")
    paths = path.rglob("*") if recursive else path.iterdir()
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def run_cuda(video: Path, output: Path, args) -> bool:
    command = [sys.executable, "-m", "subtitle_creator.engine", str(video), str(output),
               "--model", str(args.model_dir or args.model)]
    if args.language:
        command.extend(["--language", args.language])
    if args.offline:
        command.append("--offline")
    if args.cache_dir:
        command.extend(["--cache-dir", str(args.cache_dir)])
    return subprocess.run(command, check=False).returncode == 0


def publish(output: Path, destination: Path, force: bool) -> bool:
    if force:
        output.replace(destination)
    else:
        # Atomic no-clobber publication, including if another process created it.
        try:
            destination.hardlink_to(output)
        except FileExistsError:
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate local SRT subtitles for MKV/MP4 files.")
    parser.add_argument("path", type=Path, help="Video file or folder of videos")
    parser.add_argument("--recursive", "-r", action="store_true", help="Also scan subfolders")
    parser.add_argument("--model", choices=("small", "medium", "large-v3"), default="small")
    parser.add_argument("--model-dir", type=Path, help="Use a local CTranslate2 Whisper model directory")
    parser.add_argument("--language", help="Spoken language code, e.g. en, sl, de (default: detect)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="CUDA failures automatically retry on CPU (default: auto)")
    parser.add_argument("--force", action="store_true", help="Replace existing SRT files")
    parser.add_argument("--offline", action="store_true", help="Use cached/local models; never download")
    parser.add_argument("--cache-dir", type=Path, help="Model download/cache directory")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        videos = discover_videos(args.path.expanduser().resolve(), args.recursive)
        if args.model_dir:
            args.model_dir = args.model_dir.expanduser().resolve()
            if not args.model_dir.is_dir():
                raise ValueError(f"Model directory does not exist: {args.model_dir}")
        if args.cache_dir:
            args.cache_dir = args.cache_dir.expanduser().resolve()
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not videos:
        print("No MKV or MP4 videos found.", file=sys.stderr)
        return 0

    # movie.mkv and movie.mp4 would both target movie.srt. Reject that ambiguity.
    destinations: set[Path] = set()
    for video in videos:
        destination = video.with_suffix(".srt")
        if destination in destinations:
            parser.error(f"Multiple videos would write {destination}; rename one video")
        destinations.add(destination)

    pending = [v for v in videos if args.force or not v.with_suffix(".srt").exists()]
    skipped = len(videos) - len(pending)
    for video in videos:
        if not args.force and video.with_suffix(".srt").exists():
            print(f"Skipping existing subtitles: {video.with_suffix('.srt')}")
    if not pending:
        return 0

    use_cuda = args.device == "cuda" or (args.device == "auto" and cuda_available())
    cpu_model = None
    completed = failed = 0
    try:
        for index, video in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {video}", flush=True)
            destination = video.with_suffix(".srt")
            try:
                with tempfile.TemporaryDirectory(prefix=".subtitle-", dir=video.parent) as directory:
                    output = Path(directory) / "result.srt"
                    gpu_done = use_cuda and run_cuda(video, output, args)
                    if use_cuda and not gpu_done:
                        print("CUDA unavailable or failed; retrying on CPU (int8).", file=sys.stderr)
                        use_cuda = False
                    if not gpu_done:
                        audio = engine.load_audio(video)
                        if cpu_model is None:
                            cpu_model = engine.create_model(
                                str(args.model_dir or args.model), "cpu", args.offline,
                                str(args.cache_dir) if args.cache_dir else None,
                            )
                        count = engine.write_transcription(cpu_model, audio, output, args.language)
                        if not count:
                            print("No speech detected; writing an empty SRT.", file=sys.stderr)
                    if publish(output, destination, args.force):
                        completed += 1
                        print(f"Saved: {destination}", flush=True)
                    else:
                        skipped += 1
                        print(f"Skipping subtitles created during transcription: {destination}")
            except Exception as error:
                failed += 1
                print(f"Failed: {video}: {error}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrupted. Incomplete subtitles discarded.", file=sys.stderr)
        return 130
    print(f"Finished: {completed} created, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0
