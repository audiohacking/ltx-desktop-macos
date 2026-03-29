# Migration to ltx-2-mlx Library

**Date:** 2026-03-29
**Status:** Approved

## Goal

Replace ~4400 LOC of vendored model code (`engine/ltx23_model/`) and custom inference logic with the `ltx-core-mlx` and `ltx-pipelines-mlx` packages from https://github.com/dgrauet/ltx-2-mlx. Keep subprocess isolation for crash safety. Gain real retake/extend/A2V/keyframe pipelines for free.

## Model Repository Changes

Repos renamed and expanded:
| Repo ID | Quantization | Contents |
|---------|-------------|----------|
| `dgrauet/ltx-2.3-mlx` | bf16 | transformer-dev, transformer-distilled, distilled LoRA, VAE, audio, vocoder |
| `dgrauet/ltx-2.3-mlx-q8` | int8 | same layout |
| `dgrauet/ltx-2.3-mlx-q4` | int4 | same layout |

Key file renames:
- `transformers.safetensors` → `transformer-distilled.safetensors`
- New: `transformer-dev.safetensors` (full 30-step model)
- New: distilled LoRA weights (for two-stage pipeline Stage 2)

## Architecture

```
SwiftUI → HTTP/WS :8000 → FastAPI (main.py)
  → mlx_runner.py (async subprocess orchestrator)
    → python -m engine.generate_v23 (single subprocess)
      → ltx_pipelines_mlx pipelines (library handles memory staging)
```

Single subprocess per generation (not two). The library's `low_memory=True` handles text encoder staging internally.

## What Is Deleted

| Path | Reason |
|------|--------|
| `engine/ltx23_model/` (18 files) | Replaced by `ltx-core-mlx` |
| `engine/encode_text_subprocess.py` | Library handles text encoding |
| `engine/prompt_enhancer.py` | Qwen removed, Gemma handles text |
| `scripts/convert_ltx23.py` | No longer converting weights manually |
| `scripts/validate_vocoder.py` | Vocoder now in library |

## What Is Rewritten

### `generate_v23.py` (~150 LOC)

New role: CLI entry point that maps args to the correct pipeline.

```python
# Pseudocode
def main():
    args = parse_args()

    # Select pipeline based on mode
    if args.mode == "t2v":
        pipeline = TextToVideoPipeline(args.model_dir, low_memory=True)
    elif args.mode == "i2v":
        pipeline = ImageToVideoPipeline(args.model_dir, low_memory=True)
    elif args.mode == "retake":
        pipeline = RetakePipeline(args.model_dir, low_memory=True)
    elif args.mode == "extend":
        pipeline = ExtendPipeline(args.model_dir, low_memory=True)
    elif args.mode == "a2v":
        pipeline = AudioToVideoPipeline(args.model_dir, low_memory=True)

    # LoRA args passed to pipeline
    pipeline.load(lora_paths=args.lora)

    # Progress reporting via stderr (same protocol)
    _progress("STATUS:Loading model")
    pipeline.generate_and_save(
        prompt=args.prompt,
        output_path=args.output_path,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        num_steps=args.num_steps,
    )
    _progress("STATUS:Done")
```

### `mlx_runner.py`

Simplified:
- Remove `_run_text_encoding_subprocess()` — no longer needed
- Single subprocess call to `python -m engine.generate_v23`
- Pass `--mode t2v|i2v|retake|extend|a2v` arg
- For retake/extend: pass `--source-video` and frame range args
- Progress parsing stays identical (STAGE/STEP/STATUS/MEMORY/PREVIEW)

### `engine/pipelines/*.py`

Thin async wrappers remain. Each maps FastAPI request params to `mlx_runner.run_mlx_generation()` kwargs. Add new `mode` parameter.

- `text_to_video.py` — passes `mode="t2v"`
- `image_to_video.py` — passes `mode="i2v"` + image path
- `preview.py` — passes `mode="t2v"` with low res/frames
- `retake.py` — passes `mode="retake"` + source video + frame range (real inference now)
- `extend.py` — passes `mode="extend"` + source video + direction (real inference now)

### `model_download_manager.py`

Update model registry:
- Repo IDs: `dgrauet/ltx-2.3-mlx`, `dgrauet/ltx-2.3-mlx-q8`, `dgrauet/ltx-2.3-mlx-q4`
- Old IDs (`ltx-2.3-mlx-distilled-*`) mapped to new ones or removed

### `lora_manager.py`

Adapt to pass LoRA paths to the pipeline. The library's `fuse_loras()` handles weight merging. No custom LoRA application code needed.

## Prompt Enhancement — Gemma replaces Qwen

`engine/prompt_enhancer.py` (Qwen3.5-2B) is deleted. The library provides Gemma-based enhancement via `GemmaLanguageModel.enhance_t2v()` / `enhance_i2v()` in `ltx_core_mlx.text_encoders.gemma`.

This is better: same model used for encoding and enhancement (no extra 1.2GB Qwen load). The library handles load/unload internally.

### Route adaptation in `main.py`
- `POST /api/v1/prompt/enhance` stays but calls a new subprocess that uses the library's enhance function
- Remove `PromptEnhancer` class import, replace with library call
- `generate_v23.py` accepts `--enhance-prompt` flag → calls `pipeline` with enhancement enabled

## What Is Unchanged

- `main.py` — route handlers, request/response models (except prompt enhance removal)
- `memory_manager.py` — server-side monitoring
- `ffmpeg_utils.py` — ffmpeg helpers
- `model_manager.py` — stub interface
- All SwiftUI frontend code (enhance button becomes no-op or hidden)
- WebSocket progress protocol
- History, presets, job queue

## Dependencies

### Add to `pyproject.toml`
```toml
[project.dependencies]
ltx-core-mlx = { git = "https://github.com/dgrauet/ltx-2-mlx", subdirectory = "packages/ltx-core-mlx" }
ltx-pipelines-mlx = { git = "https://github.com/dgrauet/ltx-2-mlx", subdirectory = "packages/ltx-pipelines-mlx" }
```

### Remove
- `mlx-video-with-audio` (or `mlx_video`) — replaced by above
- `mlx-lm` — was only used for Qwen3.5-2B prompt enhancer, no longer needed

## Progress Reporting

The library pipelines don't expose step-level callbacks. Strategy:

1. **STATUS messages** — emitted by `generate_v23.py` before/after each phase (load, generate, decode, save)
2. **MEMORY messages** — emitted using `memory_manager.get_memory_stats()` between phases
3. **Step-level progress** — investigate if pipelines can be subclassed or monkeypatched to emit per-step callbacks. Fallback: report only stage-level progress.
4. **PREVIEW frames** — requires hooking into the denoising loop. If library doesn't support it, progressive preview is deferred.

## New Capabilities Unlocked

| Feature | Before | After |
|---------|--------|-------|
| Retake | Stub (solid color + sleep) | Real temporal-masked reinference |
| Extend | Stub (solid color + sleep) | Real frame extension |
| Audio-to-Video | Not available | Full A2V pipeline |
| Keyframe Interpolation | Not available | Smooth transitions between images |
| IC-LoRA | Not available | Control-based generation |
| Vocoder BWE | 16kHz output | 48kHz output |
| Dev model | Not supported | 30-step dev transformer available |
| Two-stage HQ | Manual upscale | Library-native two-stage pipeline |

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Library API changes | Pin to specific commit in pyproject.toml |
| Progress reporting gaps | Stage-level progress is acceptable; step-level is nice-to-have |
| Progressive preview loss | Can be re-added by subclassing pipeline's denoise loop |
| Memory regression | Library has `low_memory=True`; verify with marathon test |
| Model path resolution | Library uses same `huggingface_hub` patterns |

## Success Criteria

1. T2V generation produces same quality output as current code
2. I2V generation works with reference image
3. Marathon test passes (10 gens, 97f@768x512, no OOM)
4. Retake and extend produce real output (not stubs)
5. Audio output at 48kHz (BWE working)
6. All existing API endpoints continue to work
7. Frontend requires zero changes
