"""Unit test for canny-edge control-video extraction."""
import subprocess
from pathlib import Path

from engine.ffmpeg_utils import extract_edges


def _make_sample(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=128x128:rate=8:duration=1",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )


def test_extract_edges_produces_video(tmp_path):
    src = tmp_path / "src.mp4"
    out = tmp_path / "edges.mp4"
    _make_sample(src)

    extract_edges(str(src), str(out))

    assert out.exists() and out.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True,
    )
    assert "video" in probe.stdout
