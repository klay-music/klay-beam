#!/usr/bin/env python3
"""
bench_disk_io_ffmpeg_mp3.py
------------------------------------------------------------------
Benchmark: disk round-trip for Vorbis/Opus (libsndfile & FFmpeg) and
MP3 128‒320 kb/s encoded with FFmpeg/libmp3lame.

pip install numpy soundfile torchaudio tabulate packaging
FFmpeg (with libvorbis & libmp3lame) must be on PATH.
"""

from __future__ import annotations
import argparse, io, logging, os, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import soundfile as sf
from packaging import version

# optional torchaudio just for MP3 *decoding* fallback
try:
    import torchaudio

    HAVE_TA = True
except ImportError:
    HAVE_TA = False

# ───────── constants ─────────
MP3_KBPS = [128, 192, 256, 320]
FFMPEG_BIN = "ffmpeg"
SF_OK = version.parse(sf.__libsndfile_version__) >= version.parse("1.2.0")


# ───────── encoding helpers ─────────
def encode_ogg_sf(audio: np.ndarray, sr: int, subtype: str) -> io.BytesIO:
    if not SF_OK:
        raise RuntimeError("libsndfile < 1.2.0 cannot write Ogg safely")
    buf = io.BytesIO()
    sf.write(buf, audio.T, sr, format="OGG", subtype=subtype)
    buf.seek(0)
    return buf


def encode_vorbis_ffmpeg(audio: np.ndarray, sr: int, q: float) -> io.BytesIO:
    ch = audio.shape[0]
    raw = audio.T.astype(np.float32, copy=False).tobytes()
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        str(ch),
        "-i",
        "pipe:0",
        "-vn",
        "-c:a",
        "vorbis",
        "-strict",
        "-2",
        "-q:a",
        str(q),
        "-f",
        "ogg",
        "pipe:1",
    ]
    try:
        res = subprocess.run(
            cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr.decode(errors="ignore"))
        raise
    buf = io.BytesIO(res.stdout)
    buf.seek(0)
    return buf


def numpy_to_mp3_ffmpeg(audio: np.ndarray, sr: int, kbps: int) -> io.BytesIO:
    """Encode (channels, samples) float32 ndarray → MP3 via FFmpeg/libmp3lame."""
    ch = audio.shape[0]
    raw = audio.T.astype(np.float32, copy=False).tobytes()
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        str(ch),
        "-i",
        "pipe:0",
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        f"{kbps}k",
        "-f",
        "mp3",
        "pipe:1",
    ]
    try:
        res = subprocess.run(
            cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr.decode(errors="ignore"))
        raise
    buf = io.BytesIO(res.stdout)
    buf.seek(0)
    return buf


# ───────── decoding helpers ─────────
def read_sf(path: os.PathLike) -> None:
    sf.read(path, dtype="float32")


def read_mp3(path: os.PathLike) -> None:
    """Try soundfile first; if that fails, fall back to torchaudio (if present)."""
    try:
        sf.read(path, dtype="float32")
    except RuntimeError:
        if not HAVE_TA:
            raise
        torchaudio.load(str(path))


# ───────── benchmark ─────────
Row = Tuple[str, float, float, float]  # label, write_s, read_s, KiB


def bench(iters: int = 5, sr: int = 48_000, dur_s: float = 10.0) -> List[Row]:
    # generate test tone
    t = np.linspace(0, dur_s, int(sr * dur_s), False, dtype=np.float32)
    sweep = np.stack(
        [
            np.sin(2 * np.pi * (440 + 220 * t) * t),
            np.sin(2 * np.pi * (660 + 330 * t) * t),
        ],
        axis=0,
    )

    cases: List[
        Tuple[
            str,
            Callable[[np.ndarray, int], io.BytesIO],
            str,
            Callable[[os.PathLike], None],
        ]
    ] = [
        ("VORBIS-SF", lambda a, s: encode_ogg_sf(a, s, "VORBIS"), ".ogg", read_sf),
        ("OPUS-SF", lambda a, s: encode_ogg_sf(a, s, "OPUS"), ".ogg", read_sf),
    ]

    for q in (2.0, 5.0, 8.0):
        cases.append(
            (
                f"VORBIS-{q}",
                lambda a, s, q=q: encode_vorbis_ffmpeg(a, s, q),
                ".ogg",
                read_sf,
            )
        )

    for kbps in MP3_KBPS:
        cases.append(
            (
                f"MP3-{kbps}k",
                lambda a, s, k=kbps: numpy_to_mp3_ffmpeg(a, s, k),
                ".mp3",
                read_mp3,
            )
        )

    rows: List[Row] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for label, encode_fn, ext, read_fn in cases:
            write_times, read_times, sizes = [], [], []
            for i in range(iters):
                # encode ➜ buffer
                t0 = time.perf_counter()
                buf = encode_fn(sweep, sr)
                # buffer ➜ disk file
                fname = Path(tmpdir) / f"{label}-{i}{ext}"
                with open(fname, "wb") as f:
                    f.write(buf.getbuffer())
                write_times.append(time.perf_counter() - t0)

                sizes.append(fname.stat().st_size)

                # read back
                t1 = time.perf_counter()
                read_fn(fname)
                read_times.append(time.perf_counter() - t1)

            rows.append(
                (
                    label,
                    sum(write_times) / iters,
                    sum(read_times) / iters,
                    np.mean(sizes) / 1024.0,
                )
            )
    return rows


# ───────── table printing ─────────
def pretty(rows: List[Row]) -> None:
    try:
        from tabulate import tabulate

        print(
            tabulate(
                [(l, f"{w:5.4f}", f"{r:5.4f}", f"{k:5.1f}") for l, w, r, k in rows],
                headers=["fmt", "write s", "read s", "KiB"],
                tablefmt="github",
            )
        )
    except ImportError:
        hdr = ["fmt", "write s", "read s", "KiB"]
        print("  ".join(h.ljust(13) for h in hdr))
        print("-" * 40)
        for l, w, r, k in rows:
            print(f"{l:<13} {w:7.4f} {r:7.4f} {k:7.1f}")


# ───────── main ─────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disk IO benchmark (FFmpeg MP3)")
    parser.add_argument(
        "-n", "--iters", type=int, default=5, help="iterations per codec (default 5)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pretty(bench(args.iters))
