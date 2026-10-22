# Subtitle Creator

Generate UTF-8 SRT subtitles beside each MKV or MP4 video using faster-whisper.
Transcription runs locally on NVIDIA CUDA or CPU, with no paid API, account, or
cloud transcription service. Audio and video are never uploaded. The first run
downloads public model weights unless you use an existing model with `--offline`.

## Install on Fedora

Use Python 3.10 or newer in a virtual environment. Python 3.11–3.13 are good choices
if your Fedora default Python is too new for a dependency's binary wheels.

```bash
sudo dnf install python3 python3-pip ffmpeg-free
cd /path/to/subtitle-creator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
subtitle-creator --help
```

Audio normally decodes directly with PyAV, which bundles FFmpeg libraries.
If that fails, the CLI extracts a temporary mono 16 kHz PCM WAV using the system
`ffmpeg` executable. Fedora's `ffmpeg-free` supports a limited codec set; if your
audio codec is unsupported by both decoders, install an FFmpeg build supporting
that codec. Missing audio tracks and decoding errors are reported per file.

## Usage

```bash
# All MKV and MP4 files directly in a folder; detect each video's language
subtitle-creator "/home/me/Videos"

# Include subfolders and use a larger model
subtitle-creator "/home/me/Videos" --recursive --model large-v3

# Transcribe a single video with a known spoken language
subtitle-creator "/home/me/Videos/film.mkv" --model medium --language sl

# Explicit CPU execution and overwrite existing subtitles
subtitle-creator "/home/me/Videos" --device cpu --force

# Try NVIDIA CUDA, automatically retrying on CPU if it fails
subtitle-creator "/home/me/Videos" --device cuda

# Download/cache weights into a chosen location on the first run
subtitle-creator "/home/me/Videos" --model small --cache-dir "$HOME/.cache/subtitle-creator"

# Subsequent runs can prohibit model downloads
subtitle-creator "/home/me/Videos" --model small --cache-dir "$HOME/.cache/subtitle-creator" --offline

# Or supply an already downloaded CTranslate2-format Whisper model directory
subtitle-creator "/home/me/Videos" --model-dir /path/to/local-whisper-model --offline

# Also available without installing the console command, from this repository
python -m subtitle_creator "/home/me/Videos"
```

`movie.mkv` produces `movie.srt` in the same directory. Extensions are case
insensitive. The first audio track is used. The output stays in the spoken
language; `--language` sets that language rather than translating. Omit it for
automatic detection. Supported models are `small` (default), `medium`, and
`large-v3`; larger models need more memory and time. `--model-dir` overrides
`--model` and must contain a faster-whisper/CTranslate2 model, not an original
Whisper `.pt` checkpoint.

Existing SRT files, including empty files, are skipped unless `--force` is set.
When no speech is detected, an empty SRT is saved and a message is printed.
If two videos in the same directory share a stem (e.g. `movie.mp4` and
`movie.mkv`), rename one before running to avoid competing for the same SRT.

The CLI reports file counts, model loading, detected language and confidence,
and a progress bar measuring processed audio seconds. Initial model loading,
audio decoding, and the first recognition window can take time before the bar
advances. Silence is filtered with VAD, and timestamps map back to the original
audio timeline. Word timestamp refinement is enabled; SRT cues use Whisper's
segment start/end times rounded to milliseconds. Recognition and timing quality
depend on the recording and model; review subtitles where precise timing matters.

Output is published atomically only after successful transcription. Failed or
interrupted runs discard temporary files and preserve any previous SRT. A batch
continues after individual failures. Exit codes: `0` success (including skipped
files or an empty folder), `1` one or more failures, `2` invalid arguments/input,
`130` interrupted.

## Optional NVIDIA acceleration

CPU works without NVIDIA libraries. For CUDA, first install a working NVIDIA
driver for your Fedora system and verify it with `nvidia-smi`. The faster-whisper
GPU setup requires CUDA 12 cuBLAS and cuDNN 9. Inside the same virtual environment:

```bash
python -m pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12>=9,<10'
export LD_LIBRARY_PATH="$(python -c 'import os, nvidia.cublas, nvidia.cudnn; print(os.path.join(nvidia.cublas.__path__[0], "lib") + ":" + os.path.join(nvidia.cudnn.__path__[0], "lib"))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
subtitle-creator "/home/me/Videos" --device auto
```

Set `LD_LIBRARY_PATH` in each shell before starting Python. These pip packages
provide user-space libraries, not the NVIDIA driver. See the upstream
[faster-whisper installation guidance](https://github.com/SYSTRAN/faster-whisper#gpu)
for CUDA version compatibility and alternative installations.

`auto` checks CUDA availability and otherwise selects CPU. CUDA uses float16;
CPU uses int8. CUDA transcription runs in a separate process for each video so
even native CUDA/cuDNN crashes can trigger CPU fallback. This reloads the GPU
model for each video. After a CUDA failure (including insufficient VRAM), the
current video restarts on CPU and the rest of the batch stays on CPU, reusing
its model. CPU still needs enough RAM for the selected model. Audio is decoded
into memory, so very long recordings also increase RAM usage.

With `--offline`, missing model files cause a clear failure instead of a download.
Normal operation may download model assets from Hugging Face; all inference is
local. No cloud inference credentials are used.

## Development

```bash
python -m unittest discover -s tests -v
```

Tests cover discovery, SRT formatting, safe publication, decoding fallback,
GPU failure recovery, batch failures, and interruption without downloading models
or requiring a GPU. Fedora FFmpeg package details are available in the
[Fedora package catalog](https://packages.fedoraproject.org/pkgs/ffmpeg/ffmpeg-free/).
