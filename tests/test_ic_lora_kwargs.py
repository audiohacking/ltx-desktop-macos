"""Regression test: the IC-LoRA gen_kwargs must bind to the library signature.

IC-LoRA uses the distilled sampler (no CFG), so ICLoraPipeline.generate_and_save
does NOT accept cfg_scale/stg_scale. Passing them raises TypeError at real
generation time (not caught by import/route smoke tests). This binds the kwargs
our subprocess builds against the actual library signature.
"""

import inspect
from types import SimpleNamespace

from engine.generate_v23 import build_ic_lora_gen_kwargs


def _fake_args(image=None):
    return SimpleNamespace(
        prompt="a scene",
        output_path="/tmp/out.mp4",
        video_conditioning=["/tmp/control.mp4:1.0"],
        height=512,
        width=768,
        num_frames=97,
        fps=24,
        seed=42,
        num_steps=8,
        conditioning_strength=1.0,
        skip_stage_2=False,
        image=image,
        image_strength=1.0,
    )


def test_ic_lora_kwargs_bind_to_library_signature():
    """Every kwarg we pass must be accepted by ICLoraPipeline.generate_and_save."""
    from ltx_pipelines_mlx import ICLoraPipeline

    kwargs = build_ic_lora_gen_kwargs(_fake_args())
    sig = inspect.signature(ICLoraPipeline.generate_and_save)
    # bind against the signature (drop the bound `self`)
    unbound = sig.replace(
        parameters=[p for n, p in sig.parameters.items() if n != "self"]
    )
    unbound.bind(**kwargs)  # raises TypeError on any unexpected/missing kwarg


def test_ic_lora_kwargs_omit_cfg():
    """IC-LoRA takes no CFG/STG guidance — guard against re-adding them."""
    kwargs = build_ic_lora_gen_kwargs(_fake_args())
    assert "cfg_scale" not in kwargs
    assert "stg_scale" not in kwargs


def test_ic_lora_kwargs_image_adds_images():
    """A reference image is forwarded as an ImageConditioningInput list."""
    kwargs = build_ic_lora_gen_kwargs(_fake_args(image="/tmp/ref.jpg"))
    assert "images" in kwargs and len(kwargs["images"]) == 1
