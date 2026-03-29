# ltx-2-mlx Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ~4400 LOC of vendored model code with `ltx-core-mlx` + `ltx-pipelines-mlx` packages, keeping subprocess isolation and all existing API contracts.

**Architecture:** Single subprocess per generation calls library pipelines directly. The library's `low_memory=True` handles text encoder staging. FastAPI routes and WebSocket protocol unchanged. Prompt enhancement switches from Qwen3.5-2B to Gemma 3 12B via the library.

**Tech Stack:** Python 3.12+, FastAPI, MLX, ltx-core-mlx, ltx-pipelines-mlx, ffmpeg, uv

---

### Task 1: Update Dependencies

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Update pyproject.toml dependencies**

Replace the MLX model dependencies with the new library packages. Remove `mlx-video-with-audio` and `mlx-lm` (Qwen no longer needed, Gemma is in ltx-core-mlx). Keep `mlx-audio` for TTS.

```toml
dependencies = [
    # MLX stack
    "mlx>=0.31.0",
    "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx@main#subdirectory=packages/ltx-core-mlx",
    "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx@main#subdirectory=packages/ltx-pipelines-mlx",
    # API
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "websockets>=13.0",
    "python-multipart>=0.0.12",
    # ML / Tensor
    "numpy>=1.26.0",
    "safetensors>=0.4.0",
    "huggingface-hub>=0.26.0",
    "transformers>=4.51.0",
    # Video/Audio processing
    "opencv-python>=4.10.0",
    "tqdm>=4.66.0",
    "pillow>=10.4.0",
    "soundfile>=0.12.0",
    # TTS via Kokoro
    "mlx-audio>=0.4.1",
]
```

- [ ] **Step 2: Sync dependencies**

Run: `cd /Users/dgrauet/Work/ltx-desktop-macos/backend && ~/.local/bin/uv sync --prerelease=allow`

Expected: Successful install of ltx-core-mlx and ltx-pipelines-mlx from GitHub.

- [ ] **Step 3: Verify imports work**

Run: `cd /Users/dgrauet/Work/ltx-desktop-macos/backend && .venv/bin/python -c "from ltx_pipelines_mlx import TextToVideoPipeline, RetakePipeline, ExtendPipeline; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd /Users/dgrauet/Work/ltx-desktop-macos
git add backend/pyproject.toml
git commit -m "feat: add ltx-core-mlx and ltx-pipelines-mlx dependencies, remove mlx-video-with-audio and mlx-lm"
```

---

### Task 2: Delete Vendored Code and Obsolete Files

**Files:**
- Delete: `backend/engine/ltx23_model/` (entire directory, 18 files)
- Delete: `backend/engine/encode_text_subprocess.py`
- Delete: `backend/engine/prompt_enhancer.py`
- Delete: `backend/engine/teacache.py` (if exists)
- Delete: `scripts/convert_ltx23.py` (if exists)
- Delete: `scripts/validate_vocoder.py` (if exists)

- [ ] **Step 1: Delete vendored model directory**

Run: `rm -rf /Users/dgrauet/Work/ltx-desktop-macos/backend/engine/ltx23_model`

- [ ] **Step 2: Delete obsolete engine files**

Run: `rm -f /Users/dgrauet/Work/ltx-desktop-macos/backend/engine/encode_text_subprocess.py /Users/dgrauet/Work/ltx-desktop-macos/backend/engine/prompt_enhancer.py /Users/dgrauet/Work/ltx-desktop-macos/backend/engine/teacache.py`

- [ ] **Step 3: Delete obsolete scripts**

Run: `rm -f /Users/dgrauet/Work/ltx-desktop-macos/scripts/convert_ltx23.py /Users/dgrauet/Work/ltx-desktop-macos/scripts/validate_vocoder.py`

- [ ] **Step 4: Commit**

```bash
cd /Users/dgrauet/Work/ltx-desktop-macos
git add -A
git commit -m "refactor: delete vendored ltx23_model, encode_text_subprocess, prompt_enhancer, teacache"
```

---

### Task 3: Rewrite generate_v23.py

This is the core subprocess entry point. It receives CLI args from `mlx_runner.py`, instantiates the correct library pipeline, and emits progress on stderr.

**Files:**
- Rewrite: `backend/engine/generate_v23.py`

- [ ] **Step 1: Write the new generate_v23.py**

```python
"""LTX-2.3 generation subprocess -- delegates to ltx-pipelines-mlx.

Invoked as: python -m engine.generate_v23 --mode t2v --prompt "..." --output-path out.mp4 ...
Emits progress on stderr in the format parsed by mlx_runner.py:
  STATUS:<message>
  STAGE:<n>:STEP:<step>:<total>
  MEMORY:<label>:active=<gb>:cache=<gb>:peak=<gb>
  PREVIEW:<filepath>
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

from engine.memory_manager import aggressive_cleanup, get_memory_stats


# ---------------------------------------------------------------------------
# Progress helpers (stderr protocol for mlx_runner.py)
# ---------------------------------------------------------------------------

def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _report_memory(label: str) -> None:
    stats = get_memory_stats()
    _progress(
        f"MEMORY:{label}"
        f":active={stats['active_memory_gb']:.3f}"
        f":cache={stats['cache_memory_gb']:.3f}"
        f":peak={stats['peak_memory_gb']:.3f}"
    )


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _create_pipeline(args: argparse.Namespace):
    """Instantiate the correct library pipeline for the given mode."""
    from ltx_pipelines_mlx import (
        ExtendPipeline,
        ImageToVideoPipeline,
        RetakePipeline,
        TextToVideoPipeline,
    )

    model_dir = args.model_dir
    gemma = args.gemma or "mlx-community/gemma-3-12b-it-4bit"
    low_memory = True

    if args.mode == "retake":
        return RetakePipeline(model_dir, gemma_model_id=gemma, low_memory=low_memory)
    elif args.mode == "extend":
        return ExtendPipeline(model_dir, gemma_model_id=gemma, low_memory=low_memory)
    elif args.mode == "i2v":
        return ImageToVideoPipeline(model_dir, gemma_model_id=gemma, low_memory=low_memory)
    else:  # t2v (default)
        return TextToVideoPipeline(model_dir, gemma_model_id=gemma, low_memory=low_memory)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _run_t2v(pipeline, args: argparse.Namespace) -> None:
    """Text-to-video or image-to-video generation."""
    _progress("STATUS:Loading model")
    _report_memory("before_load")

    lora_paths = _parse_lora_args(args.lora) if args.lora else None
    pipeline.load(lora_paths=lora_paths)

    _report_memory("after_model_load")
    _progress("STATUS:Generating video")

    enhance = args.enhance_prompt

    if args.mode == "i2v" and args.image:
        output = pipeline.generate_and_save(
            prompt=args.prompt,
            output_path=args.output_path,
            image=args.image,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            num_steps=args.num_steps,
            enhance_prompt=enhance,
        )
    else:
        output = pipeline.generate_and_save(
            prompt=args.prompt,
            output_path=args.output_path,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            num_steps=args.num_steps,
            enhance_prompt=enhance,
        )

    _report_memory("after_generation")
    _progress("STATUS:Done")


def _run_retake(pipeline, args: argparse.Namespace) -> None:
    """Retake: regenerate a frame range in an existing video."""
    _progress("STATUS:Loading model")
    _report_memory("before_load")

    lora_paths = _parse_lora_args(args.lora) if args.lora else None
    pipeline.load(lora_paths=lora_paths)

    _report_memory("after_model_load")
    _progress("STATUS:Retaking segment")

    video_latent, audio_latent = pipeline.retake_from_video(
        prompt=args.prompt,
        video_path=args.retake_source,
        start_frame=args.retake_start_frame,
        end_frame=args.retake_end_frame,
        seed=args.seed,
        num_steps=args.num_steps or 30,
    )

    _progress("STATUS:Decoding video")
    pipeline._decode_and_save_video(video_latent, audio_latent, args.output_path)

    _report_memory("after_generation")
    _progress("STATUS:Done")


def _run_extend(pipeline, args: argparse.Namespace) -> None:
    """Extend: add frames before or after an existing video."""
    _progress("STATUS:Loading model")
    _report_memory("before_load")

    lora_paths = _parse_lora_args(args.lora) if args.lora else None
    pipeline.load(lora_paths=lora_paths)

    _report_memory("after_model_load")
    _progress("STATUS:Extending video")

    video_latent, audio_latent = pipeline.extend_from_video(
        prompt=args.prompt,
        video_path=args.extend_source,
        extend_frames=args.extend_frames,
        direction=args.extend_direction,
        seed=args.seed,
        num_steps=args.num_steps or 30,
    )

    _progress("STATUS:Decoding video")
    pipeline._decode_and_save_video(video_latent, audio_latent, args.output_path)

    _report_memory("after_generation")
    _progress("STATUS:Done")


# ---------------------------------------------------------------------------
# LoRA arg parsing
# ---------------------------------------------------------------------------

def _parse_lora_args(lora_list: list[str]) -> list[tuple[str, float]]:
    """Parse --lora path:strength args into list of (path, strength) tuples."""
    result = []
    for entry in lora_list:
        # Handle macOS paths with colons -- use rfind
        idx = entry.rfind(":")
        if idx > 0 and idx < len(entry) - 1:
            try:
                strength = float(entry[idx + 1:])
                path = entry[:idx]
                result.append((path, strength))
                continue
            except ValueError:
                pass
        result.append((entry, 0.7))
    return result


# ---------------------------------------------------------------------------
# Prompt enhancement (standalone subprocess mode)
# ---------------------------------------------------------------------------

def _run_enhance(args: argparse.Namespace) -> None:
    """Enhance a prompt using Gemma via the library text encoder."""
    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

    _progress("STATUS:Loading Gemma for enhancement")
    gemma = GemmaLanguageModel(args.gemma or "mlx-community/gemma-3-12b-it-4bit")
    gemma.load()

    if args.enhance_mode == "i2v":
        enhanced = gemma.enhance_i2v(args.prompt, seed=args.seed)
    else:
        enhanced = gemma.enhance_t2v(args.prompt, seed=args.seed)

    # Write enhanced prompt to stdout (not stderr -- stderr is for progress)
    print(enhanced, flush=True)

    del gemma
    aggressive_cleanup()
    _progress("STATUS:Done")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LTX-2.3 generation subprocess")

    parser.add_argument("--mode", choices=["t2v", "i2v", "retake", "extend", "enhance"],
                        default="t2v", help="Pipeline mode")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-dir", required=True, help="HF model path or repo ID")
    parser.add_argument("--output-path", default="output.mp4")
    parser.add_argument("--gemma", default=None, help="Gemma model ID for text encoding")

    # Video params
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-steps", type=int, default=8)

    # I2V
    parser.add_argument("--image", default=None, help="Reference image path for I2V")
    parser.add_argument("--image-strength", type=float, default=1.0)

    # Retake
    parser.add_argument("--retake-source", default=None, help="Source video for retake")
    parser.add_argument("--retake-start-frame", type=int, default=0)
    parser.add_argument("--retake-end-frame", type=int, default=-1)

    # Extend
    parser.add_argument("--extend-source", default=None, help="Source video for extend")
    parser.add_argument("--extend-frames", type=int, default=49)
    parser.add_argument("--extend-direction", choices=["before", "after"], default="after")

    # LoRA
    parser.add_argument("--lora", action="append", default=None,
                        help="LoRA path:strength (can repeat)")

    # Enhancement
    parser.add_argument("--enhance-prompt", action="store_true",
                        help="Enhance prompt via Gemma before generation")
    parser.add_argument("--enhance-mode", choices=["t2v", "i2v"], default="t2v")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.mode == "enhance":
        _run_enhance(args)
        return

    pipeline = _create_pipeline(args)

    if args.mode == "retake":
        _run_retake(pipeline, args)
    elif args.mode == "extend":
        _run_extend(pipeline, args)
    else:
        _run_t2v(pipeline, args)

    # Final cleanup
    del pipeline
    aggressive_cleanup()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the module can be parsed**

Run: `cd /Users/dgrauet/Work/ltx-desktop-macos/backend && .venv/bin/python -c "import ast; ast.parse(open('engine/generate_v23.py').read()); print('syntax OK')"`

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/dgrauet/Work/ltx-desktop-macos
git add backend/engine/generate_v23.py
git commit -m "refactor: rewrite generate_v23.py to use ltx-pipelines-mlx library"
```

---

### Task 4: Rewrite mlx_runner.py

Simplify to a single subprocess (no separate text encoding). Update model repo IDs. Keep stderr progress parsing intact.

**Files:**
- Rewrite: `backend/engine/mlx_runner.py`

- [ ] **Step 1: Write the new mlx_runner.py**

```python
"""Async subprocess orchestrator for MLX generation.

Launches ``python -m engine.generate_v23`` as a subprocess and parses its
stderr output for real-time progress reporting back to the FastAPI server.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_REPO = "dgrauet/ltx-2.3-mlx-q8"

_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Stderr line patterns (must match generate_v23.py output)
_STAGE_RE = re.compile(r"^STAGE:(\d+):STEP:(\d+):(\d+)")
_STATUS_RE = re.compile(r"^STATUS:(.+)")
_MEMORY_RE = re.compile(r"^MEMORY:(\w+):active=([\d.]+):cache=([\d.]+):peak=([\d.]+)")
_PREVIEW_RE = re.compile(r"^PREVIEW:(.+)")

# Progress ranges for mapping STAGE lines to 0.0-1.0
_STAGE_RANGES = {1: (0.05, 0.55), 2: (0.65, 0.80)}
_SINGLE_STAGE_RANGES = {1: (0.05, 0.85)}

# Status string -> approximate progress
_STATUS_PROGRESS = {
    "loading": 0.02,
    "stage 1": 0.06,
    "generating": 0.10,
    "upscaling latent": 0.57,
    "reloading model": 0.60,
    "stage 2": 0.63,
    "upscaling": 0.83,
    "decoding video": 0.88,
    "decoding audio": 0.93,
    "retaking": 0.10,
    "extending": 0.10,
    "saving": 0.97,
    "done": 1.0,
}


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def get_model_repo(repo_id: str | None = None) -> tuple[str, bool]:
    """Resolve a HuggingFace repo ID to a local cache path.

    Returns:
        (model_path, is_quantized) -- path is the repo ID if not cached locally.
    """
    target_repo = repo_id or DEFAULT_MODEL_REPO

    model_path = _resolve_hf_model(target_repo)
    if model_path:
        quantized = _is_quantized_model(Path(model_path))
        log.info("Using HF model: %s (%s)", target_repo, model_path)
        return model_path, quantized

    log.warning("Could not resolve model %s -- returning repo ID", target_repo)
    return target_repo, True


def _resolve_hf_model(repo_id: str) -> str | None:
    """Check HF cache for a downloaded model. Returns directory path or None."""
    for check_file in ("transformer-distilled.safetensors", "transformer-dev.safetensors"):
        result = try_to_load_from_cache(repo_id, check_file)
        if result and isinstance(result, str):
            return str(Path(result).parent)
    return None


def _is_quantized_model(model_dir: Path) -> bool:
    """Check if model has quantization config."""
    return (model_dir / "quantize_config.json").exists()


def get_venv_python() -> str:
    """Auto-detect the venv Python binary."""
    backend_dir = Path(__file__).resolve().parent.parent
    venv_python = backend_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    raise FileNotFoundError(f"No venv Python at {venv_python}")


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _compute_progress(stage: int, step: int, total: int, *, two_stage: bool = False) -> float:
    """Map (stage, step, total) to a 0.0-1.0 progress value."""
    ranges = _STAGE_RANGES if two_stage else _SINGLE_STAGE_RANGES
    lo, hi = ranges.get(stage, (0.0, 1.0))
    if total <= 0:
        return lo
    frac = step / total
    return lo + frac * (hi - lo)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

async def run_mlx_generation(
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    fps: int,
    output_path: str,
    mode: str = "t2v",
    image: str | None = None,
    image_strength: float = 1.0,
    num_steps: int = 8,
    enhance_prompt: bool = False,
    lora_args: list[str] | None = None,
    retake_source: str | None = None,
    retake_start_frame: int = 0,
    retake_end_frame: int = -1,
    extend_source: str | None = None,
    extend_frames: int = 49,
    extend_direction: str = "after",
    progress_callback: Callable[..., Awaitable[None]] | None = None,
    venv_python: str | None = None,
    model_repo_id: str | None = None,
) -> dict:
    """Launch a generation subprocess and stream progress back.

    Returns:
        Dict with ``output_path`` and ``subprocess_memory`` snapshots.
    """
    python_bin = venv_python or get_venv_python()
    model_repo, _ = get_model_repo(model_repo_id)
    backend_dir = str(Path(__file__).resolve().parent.parent)

    # Build command
    cmd = [
        python_bin, "-m", "engine.generate_v23",
        "--mode", mode,
        "--prompt", prompt,
        "--model-dir", model_repo,
        "--height", str(height),
        "--width", str(width),
        "--num-frames", str(num_frames),
        "--seed", str(seed),
        "--fps", str(fps),
        "--output-path", output_path,
        "--num-steps", str(num_steps),
    ]

    # I2V args
    if image:
        cmd.extend(["--image", image, "--image-strength", str(image_strength)])

    # Retake args
    if retake_source:
        cmd.extend([
            "--retake-source", retake_source,
            "--retake-start-frame", str(retake_start_frame),
            "--retake-end-frame", str(retake_end_frame),
        ])

    # Extend args
    if extend_source:
        cmd.extend([
            "--extend-source", extend_source,
            "--extend-frames", str(extend_frames),
            "--extend-direction", extend_direction,
        ])

    # LoRA args
    if lora_args:
        for la in lora_args:
            cmd.extend(["--lora", la])

    # Prompt enhancement
    if enhance_prompt:
        cmd.append("--enhance-prompt")
        if mode == "i2v":
            cmd.extend(["--enhance-mode", "i2v"])

    # Launch subprocess
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=backend_dir,
        env=env,
    )

    # Parse stderr for progress
    subprocess_memory: dict[str, dict] = {}
    last_pct = 0.0
    last_step, last_total, last_stage = 0, 0, 1

    assert proc.stderr is not None
    while True:
        line_bytes = await proc.stderr.readline()
        if not line_bytes:
            break
        line = line_bytes.decode("utf-8", errors="replace").rstrip()

        # PREVIEW frame
        m = _PREVIEW_RE.match(line)
        if m:
            fpath = m.group(1).strip()
            try:
                with open(fpath, "rb") as f:
                    b64_frame = base64.b64encode(f.read()).decode("ascii")
                os.unlink(fpath)
                if progress_callback:
                    r = progress_callback(
                        last_step, last_total, last_pct, b64_frame, status=None,
                    )
                    if asyncio.iscoroutine(r):
                        await r
            except Exception:
                log.debug("Failed to read preview frame: %s", fpath)
            continue

        # STAGE/STEP progress
        m = _STAGE_RE.match(line)
        if m:
            stage, step, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
            pct = _compute_progress(stage, step, total)
            last_pct, last_step, last_total, last_stage = pct, step, total, stage
            if progress_callback:
                r = progress_callback(step, total, pct, None, status="Generating video")
                if asyncio.iscoroutine(r):
                    await r
            continue

        # MEMORY snapshot
        m = _MEMORY_RE.match(line)
        if m:
            label = m.group(1)
            subprocess_memory[label] = {
                "active_memory_gb": float(m.group(2)),
                "cache_memory_gb": float(m.group(3)),
                "peak_memory_gb": float(m.group(4)),
            }
            log.info("MEMORY[%s] active=%.1fGB cache=%.1fGB peak=%.1fGB",
                     label, float(m.group(2)), float(m.group(3)), float(m.group(4)))
            continue

        # STATUS message
        m = _STATUS_RE.match(line)
        if m:
            status_msg = m.group(1).strip()
            status_lower = status_msg.lower()
            for key, pct_val in _STATUS_PROGRESS.items():
                if key in status_lower:
                    last_pct = pct_val
                    break
            if progress_callback:
                r = progress_callback(last_step, last_total, last_pct, None, status=status_msg)
                if asyncio.iscoroutine(r):
                    await r
            continue

        # Other stderr lines -> log
        if line:
            log.debug("subprocess: %s", line[-200:])

    await proc.wait()

    if proc.returncode != 0:
        remaining = await proc.stderr.read()
        error_tail = remaining.decode("utf-8", errors="replace")[-500:] if remaining else ""
        if proc.returncode == -6:
            raise RuntimeError(f"GPU out of memory (exit code -6). {error_tail}")
        raise RuntimeError(
            f"Generation subprocess failed (exit {proc.returncode}). {error_tail}"
        )

    return {
        "output_path": output_path,
        "subprocess_memory": subprocess_memory,
    }


# ---------------------------------------------------------------------------
# Prompt enhancement (subprocess)
# ---------------------------------------------------------------------------

async def run_prompt_enhance(
    prompt: str,
    is_i2v: bool = False,
    model_repo_id: str | None = None,
    venv_python: str | None = None,
) -> str:
    """Run prompt enhancement in a subprocess via Gemma.

    Returns the enhanced prompt string.
    """
    python_bin = venv_python or get_venv_python()
    model_repo, _ = get_model_repo(model_repo_id)
    backend_dir = str(Path(__file__).resolve().parent.parent)

    cmd = [
        python_bin, "-m", "engine.generate_v23",
        "--mode", "enhance",
        "--prompt", prompt,
        "--model-dir", model_repo,
        "--enhance-mode", "i2v" if is_i2v else "t2v",
        "--seed", "10",
    ]

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=backend_dir,
        env=env,
    )

    stdout_bytes, stderr_bytes = await proc.communicate()

    if proc.returncode != 0:
        error = stderr_bytes.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"Prompt enhancement failed (exit {proc.returncode}). {error}")

    return stdout_bytes.decode("utf-8").strip()
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/dgrauet/Work/ltx-desktop-macos/backend && .venv/bin/python -c "import ast; ast.parse(open('engine/mlx_runner.py').read()); print('syntax OK')"`

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/dgrauet/Work/ltx-desktop-macos
git add backend/engine/mlx_runner.py
git commit -m "refactor: simplify mlx_runner.py -- single subprocess, remove text encoding split"
```

---

### Task 5: Update Model Download Manager

Update model IDs and repo names to match the renamed HuggingFace repos.

**Files:**
- Modify: `backend/engine/model_download_manager.py`

- [ ] **Step 1: Read the current file to find _KNOWN_MODELS**

Read `backend/engine/model_download_manager.py` and locate the `_KNOWN_MODELS` list.

- [ ] **Step 2: Update _KNOWN_MODELS registry**

Replace model entries with updated repo names and check files. Remove Qwen3.5-2B entry.

New entries:
```python
_KNOWN_MODELS = [
    {
        "id": "ltx-2.3-mlx-q8",
        "name": "LTX-2.3 (int8)",
        "size_gb": 28.0,
        "model_type": "video_generator",
        "hf_repo": "dgrauet/ltx-2.3-mlx-q8",
        "check_file": "transformer-distilled.safetensors",
    },
    {
        "id": "ltx-2.3-mlx-q4",
        "name": "LTX-2.3 (int4)",
        "size_gb": 15.0,
        "model_type": "video_generator",
        "hf_repo": "dgrauet/ltx-2.3-mlx-q4",
        "check_file": "transformer-distilled.safetensors",
    },
    {
        "id": "ltx-2.3-mlx",
        "name": "LTX-2.3 (bf16)",
        "size_gb": 42.0,
        "model_type": "video_generator",
        "hf_repo": "dgrauet/ltx-2.3-mlx",
        "check_file": "transformer-distilled.safetensors",
    },
    {
        "id": "gemma-3-12b-it-4bit",
        "name": "Gemma 3 12B IT (4-bit)",
        "size_gb": 6.0,
        "model_type": "text_encoder",
        "hf_repo": "mlx-community/gemma-3-12b-it-4bit",
        "check_file": "model.safetensors.index.json",
    },
]
```

- [ ] **Step 3: Update DEFAULT_MODEL_REPO if present**

Search for any default repo ID references and update to `dgrauet/ltx-2.3-mlx-q8`.

- [ ] **Step 4: Commit**

```bash
cd /Users/dgrauet/Work/ltx-desktop-macos
git add backend/engine/model_download_manager.py
git commit -m "refactor: update model registry -- renamed repos, remove Qwen entry"
```

---

### Task 6: Update Pipeline Wrappers

Each pipeline wrapper is a thin async layer between FastAPI and `mlx_runner`. Add `mode` parameter and update retake/extend to use real inference.

**Files:**
- Modify: `backend/engine/pipelines/text_to_video.py`
- Modify: `backend/engine/pipelines/image_to_video.py`
- Modify: `backend/engine/pipelines/preview.py`
- Rewrite: `backend/engine/pipelines/retake.py`
- Rewrite: `backend/engine/pipelines/extend.py`

- [ ] **Step 1: Update text_to_video.py**

Read the file. Find the call to `run_mlx_generation()`. Add `mode="t2v"`. Remove args that no longer exist in the new runner: `preview_interval`, `skip_bwe`, `upscale`, `ffmpeg_upscale`.

Key change: add `mode="t2v"` to the `run_mlx_generation()` call.

- [ ] **Step 2: Update image_to_video.py**

Read the file. Find the call to `run_mlx_generation()`. Add `mode="i2v"`. Remove obsolete args.

- [ ] **Step 3: Update preview.py**

Read the file. Add `mode="i2v" if image else "t2v"` to `run_mlx_generation()`. Remove `ffmpeg_upscale`, `preview_interval`.

- [ ] **Step 4: Rewrite retake.py -- real inference**

Replace the stub implementation with a real pipeline call:

```python
"""Retake pipeline -- regenerate a segment of an existing video."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from engine.ffmpeg_utils import probe_video_info
from engine.memory_manager import aggressive_cleanup, reset_peak_memory
from engine.mlx_runner import run_mlx_generation

log = logging.getLogger(__name__)

_VAE_TEMPORAL_FACTOR = 8
_OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs" / "retakes"


@dataclass
class GenerationResult:
    job_id: str
    output_path: str
    duration_seconds: float
    memory_after: dict
    stages: dict[str, float] = field(default_factory=dict)


def _pixel_time_to_latent_frame(time_s: float, fps: int) -> int:
    pixel_frame = int(time_s * fps)
    return pixel_frame // _VAE_TEMPORAL_FACTOR


def _round_to_vae_compatible(num_frames: int) -> int:
    k = max(1, (num_frames - 1) // _VAE_TEMPORAL_FACTOR)
    return 1 + k * _VAE_TEMPORAL_FACTOR


class RetakePipeline:
    def __init__(self, model_manager) -> None:
        self._model_manager = model_manager

    async def generate(
        self,
        source_video_path: str,
        prompt: str,
        start_time_s: float,
        end_time_s: float,
        steps: int = 8,
        seed: int = 42,
        fps: int = 24,
        model_repo_id: str | None = None,
        progress_callback=None,
    ) -> GenerationResult:
        job_id = uuid.uuid4().hex[:8]
        t0 = time.monotonic()
        aggressive_cleanup()
        reset_peak_memory()

        info = probe_video_info(source_video_path)
        width = info.get("width", 768)
        height = info.get("height", 512)
        duration = info.get("duration", 4.0)
        num_frames = int(duration * fps)
        num_frames = _round_to_vae_compatible(num_frames)

        start_frame = _pixel_time_to_latent_frame(start_time_s, fps)
        end_frame = _pixel_time_to_latent_frame(end_time_s, fps)

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(_OUTPUT_DIR / f"retake_{job_id}.mp4")

        gen_result = await run_mlx_generation(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=fps,
            output_path=output_path,
            mode="retake",
            num_steps=steps,
            retake_source=source_video_path,
            retake_start_frame=start_frame,
            retake_end_frame=end_frame,
            model_repo_id=model_repo_id,
            progress_callback=progress_callback,
        )

        aggressive_cleanup()
        elapsed = time.monotonic() - t0

        return GenerationResult(
            job_id=job_id,
            output_path=output_path,
            duration_seconds=elapsed,
            memory_after=gen_result.get("subprocess_memory", {}).get("after_generation", {}),
            stages={"retake": elapsed},
        )
```

- [ ] **Step 5: Rewrite extend.py -- real inference**

Replace the stub with a real pipeline call:

```python
"""Extend pipeline -- add frames before or after an existing video."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from engine.memory_manager import aggressive_cleanup, reset_peak_memory
from engine.mlx_runner import run_mlx_generation

log = logging.getLogger(__name__)

_OUTPUT_DIR = Path.home() / ".ltx-desktop" / "outputs" / "extensions"


@dataclass
class GenerationResult:
    job_id: str
    output_path: str
    duration_seconds: float
    memory_after: dict
    stages: dict[str, float] = field(default_factory=dict)


class ExtendPipeline:
    def __init__(self, model_manager) -> None:
        self._model_manager = model_manager

    async def generate(
        self,
        source_video_path: str,
        prompt: str,
        direction: str,
        extension_frames: int = 49,
        steps: int = 8,
        seed: int = 42,
        fps: int = 24,
        model_repo_id: str | None = None,
        progress_callback=None,
    ) -> GenerationResult:
        job_id = uuid.uuid4().hex[:8]
        t0 = time.monotonic()
        aggressive_cleanup()
        reset_peak_memory()

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(_OUTPUT_DIR / f"extend_{job_id}.mp4")

        # Map "forward"/"backward" to library "after"/"before"
        lib_direction = "after" if direction == "forward" else "before"

        gen_result = await run_mlx_generation(
            prompt=prompt,
            height=0,
            width=0,
            num_frames=extension_frames,
            seed=seed,
            fps=fps,
            output_path=output_path,
            mode="extend",
            num_steps=steps,
            extend_source=source_video_path,
            extend_frames=extension_frames,
            extend_direction=lib_direction,
            model_repo_id=model_repo_id,
            progress_callback=progress_callback,
        )

        aggressive_cleanup()
        elapsed = time.monotonic() - t0

        return GenerationResult(
            job_id=job_id,
            output_path=output_path,
            duration_seconds=elapsed,
            memory_after=gen_result.get("subprocess_memory", {}).get("after_generation", {}),
            stages={"extend": elapsed},
        )
```

- [ ] **Step 6: Commit**

```bash
cd /Users/dgrauet/Work/ltx-desktop-macos
git add backend/engine/pipelines/
git commit -m "refactor: update pipeline wrappers -- use mode param, real retake/extend"
```

---

### Task 7: Update main.py -- Prompt Enhancement Route

Replace the Qwen-based prompt enhancement with the new subprocess-based Gemma enhancement.

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Read main.py and locate prompt enhancer references**

Find all occurrences of `PromptEnhancer`, `prompt_enhancer`, and the enhance endpoint.

- [ ] **Step 2: Remove PromptEnhancer import and global instance**

Remove the import line:
```python
from engine.prompt_enhancer import PromptEnhancer
```
Remove the global instance:
```python
prompt_enhancer = PromptEnhancer()
```

- [ ] **Step 3: Add import for new enhancement function**

Add to the imports section near other engine imports:
```python
from engine.mlx_runner import run_prompt_enhance
```

- [ ] **Step 4: Rewrite the enhance_prompt endpoint**

Replace the endpoint body with:
```python
@app.post("/api/v1/prompt/enhance")
async def enhance_prompt(req: EnhanceRequest):
    """Enhance a prompt using Gemma 3 12B via subprocess."""
    try:
        enhanced = await run_prompt_enhance(
            prompt=req.prompt,
            is_i2v=req.is_i2v,
            model_repo_id=selected_video_model,
        )
        return EnhanceResponse(original=req.prompt, enhanced=enhanced)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {e}")
```

- [ ] **Step 5: Verify syntax**

Run: `cd /Users/dgrauet/Work/ltx-desktop-macos/backend && .venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('syntax OK')"`

Expected: `syntax OK`

- [ ] **Step 6: Commit**

```bash
cd /Users/dgrauet/Work/ltx-desktop-macos
git add backend/main.py
git commit -m "refactor: replace Qwen prompt enhancer with Gemma via subprocess"
```

---

### Task 8: Update engine/__init__.py and Cleanup Imports

Remove references to deleted modules from the engine package init and any other files.

**Files:**
- Modify: `backend/engine/__init__.py`
- Modify: `backend/audio/tts_engine.py` (verify no broken imports)

- [ ] **Step 1: Read and update engine/__init__.py**

Read the file. Remove any imports referencing `ltx23_model`, `encode_text_subprocess`, `PromptEnhancer`, or `TeaCache`. Keep exports for `ModelManager`, pipeline classes, `aggressive_cleanup`, `run_mlx_generation`.

- [ ] **Step 2: Verify no remaining broken imports**

Run: `cd /Users/dgrauet/Work/ltx-desktop-macos/backend && .venv/bin/python -c "import engine; print('engine OK')"`

Expected: `engine OK`

Run: `cd /Users/dgrauet/Work/ltx-desktop-macos/backend && .venv/bin/python -c "from engine.mlx_runner import run_mlx_generation; print('runner OK')"`

Expected: `runner OK`

- [ ] **Step 3: Verify main.py imports cleanly**

Run: `cd /Users/dgrauet/Work/ltx-desktop-macos/backend && .venv/bin/python -c "import main; print('main OK')"`

Expected: `main OK`

- [ ] **Step 4: Commit**

```bash
cd /Users/dgrauet/Work/ltx-desktop-macos
git add backend/engine/__init__.py backend/audio/
git commit -m "refactor: cleanup engine imports -- remove references to deleted modules"
```

---

### Task 9: Update CLAUDE.md Files

Update project documentation to reflect the new architecture.

**Files:**
- Modify: `/Users/dgrauet/Work/ltx-desktop-macos/CLAUDE.md`
- Modify: `/Users/dgrauet/Work/ltx-desktop-macos/backend/CLAUDE.md`

- [ ] **Step 1: Update root CLAUDE.md**

Key changes:
- Replace `mlx-video-with-audio` with `ltx-core-mlx` + `ltx-pipelines-mlx`
- Remove Qwen3.5-2B / mlx-lm references for prompt enhancement, note Gemma-based enhancement
- Update model repo IDs: `dgrauet/ltx-2.3-mlx-q8`, `dgrauet/ltx-2.3-mlx-q4`, `dgrauet/ltx-2.3-mlx`
- Note `transformer.safetensors` renamed to `transformer-distilled.safetensors`
- Update architecture: single subprocess, library handles text encoder staging
- Remove encode_text_subprocess from architecture diagram
- Update Features: retake/extend now real, audio BWE at 48kHz, A2V/keyframe available
- Remove TeaCache section (library's domain now)
- Remove detailed vendored model architecture section (now in library)

- [ ] **Step 2: Update backend/CLAUDE.md**

Key changes:
- Remove `engine/ltx23_model/` from directory structure
- Remove `encode_text_subprocess.py`, `prompt_enhancer.py`, `teacache.py` from directory listing
- Update dependencies: replace `mlx-video-with-audio` and `mlx-lm` with library packages
- Remove `mlx-lm` from key packages
- Note: model loading handled by library pipelines

- [ ] **Step 3: Commit**

```bash
cd /Users/dgrauet/Work/ltx-desktop-macos
git add CLAUDE.md backend/CLAUDE.md
git commit -m "docs: update CLAUDE.md -- reflect ltx-2-mlx library migration"
```

---

### Task 10: Integration Test -- Server Startup

Verify the server starts and basic endpoints respond.

**Files:** None (testing only)

- [ ] **Step 1: Start the backend**

Run: `cd /Users/dgrauet/Work/ltx-desktop-macos/backend && timeout 10 .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000 &`

Wait 3 seconds, then:

Run: `curl -s http://127.0.0.1:8000/api/v1/system/health`

Expected: `{"status":"ok"}`

- [ ] **Step 2: Test model list endpoint**

Run: `curl -s http://127.0.0.1:8000/api/v1/models`

Expected: JSON with updated model IDs (`ltx-2.3-mlx-q8`, etc.), no Qwen entry.

- [ ] **Step 3: Test prompt enhancement endpoint**

Run: `curl -s -X POST http://127.0.0.1:8000/api/v1/prompt/enhance -H 'Content-Type: application/json' -d '{"prompt": "a cat walking"}'`

Expected: JSON with `original` and `enhanced` fields. May return HTTP 500 if model not downloaded -- that is acceptable, verify error is clean not a server crash.

- [ ] **Step 4: Stop server**

Kill the background server process.

- [ ] **Step 5: Fix any issues found, re-test**

If any endpoint fails with import errors or crashes, fix and re-test.

---

### Task 11: End-to-End Generation Test

Run a real T2V generation to verify the full pipeline works.

**Files:** None (testing only)

- [ ] **Step 1: Start server and run T2V generation**

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/generate/text-to-video \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A golden retriever playing in a sunlit park, green grass, warm afternoon light",
    "width": 384,
    "height": 256,
    "num_frames": 9,
    "steps": 4,
    "seed": 7
  }'
```

Expected: JSON response with `job_id`.

- [ ] **Step 2: Poll job status until completion**

Run: `curl -s http://127.0.0.1:8000/api/v1/queue/<JOB_ID>`

Expected: Eventually `"status": "completed"` with valid `output_path`.

- [ ] **Step 3: Verify output video**

Run: `ffprobe -v error -show_format -show_streams <OUTPUT_PATH> 2>&1 | head -20`

Expected: Video stream present, audio stream present, reasonable duration.

- [ ] **Step 4: Fix any issues and re-test**

Common issues:
- Import path mismatches between generate_v23.py and library
- Missing argument format (local path vs repo ID)
- LoRA argument format differences
- Audio decode/muxing issues

Stop server after all tests pass.

---

### Task 12: Update Memory File

Update the project memory to reflect the migration.

**Files:**
- Modify: `/Users/dgrauet/.claude/projects/-Users-dgrauet-Work-ltx-desktop-macos/memory/MEMORY.md`
- Create or update relevant memory files

- [ ] **Step 1: Update memory entries**

Key updates:
- Model repo IDs changed to `dgrauet/ltx-2.3-mlx-q8`, `q4`, `mlx`
- Architecture: single subprocess (not two), library handles text encoder staging
- Prompt enhancement: Gemma via library, not Qwen
- `ltx23_model/` deleted, replaced by `ltx-core-mlx` + `ltx-pipelines-mlx` packages
- Retake/extend are real implementations now
- Audio BWE included in library vocoder (48kHz output)
- `transformer.safetensors` renamed to `transformer-distilled.safetensors`
- `mlx-video-with-audio` and `mlx-lm` removed from dependencies
