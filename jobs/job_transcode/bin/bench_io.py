#!/usr/bin/env python3
"""
bench_ogg.py
------------------------------------------------------------------
Benchmark in-memory Ogg encoding (Vorbis & Opus) + decoding
with soundfile/libsndfile ≥ 1.2.0.

Dependencies
------------
pip install soundfile numpy tabulate packaging
"""

from __future__ import annotations
import argparse, io, logging, time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from packaging import version

# ────────── your numpy_to_ogg helper (slightly parameterised) ──────────
SF_REQUIRED = version.parse("1.2.0")
SF_OK = version.parse(sf.__libsndfile_version__) >= SF_REQUIRED


def numpy_to_ogg(
    audio_data: np.ndarray,
    sr: int,
    *,
    safe: bool = True,
    subtype: str = "VORBIS",
) -> io.BytesIO:
    """
    Encode (channels, samples) float32 ndarray to Ogg/<subtype> in memory.
    subtype: "VORBIS" or "OPUS"
    Returns a BytesIO positioned at 0.
    """
    assert audio_data.ndim == 2, "audio_data must be 2-D (channels, samples)"
    if not SF_OK and safe:
        raise RuntimeError(
            f"libsndfile {sf.__libsndfile_version__} < 1.2.0; "
            "in-memory Ogg write is unsafe."
        )

    buf = io.BytesIO()
    sf.write(buf, audio_data.T, sr, format="OGG", subtype=subtype)
    buf.seek(0)
    return buf


# ───────────────────────────── benchmark ──────────────────────────────
Row = Tuple[str, float, float, float]  # fmt, write_s, read_s, KiB


def bench(iters: int = 5, sr: int = 48_000, dur_s: float = 10.0) -> List[Row]:
    """Return list of rows for Vorbis and Opus."""
    # test signal: stereo sine sweep
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False, dtype=np.float32)
    sweep = np.stack(
        [
            np.sin(2 * np.pi * (440 + 220 * t) * t),
            np.sin(2 * np.pi * (660 + 330 * t) * t),
        ],
        axis=0,
    )  # (ch, n)

    rows: List[Row] = []
    for subtype in ("VORBIS", "OPUS"):
        # --- write ---
        sizes = []
        t0 = time.perf_counter()
        for _ in range(iters):
            buf = numpy_to_ogg(sweep, sr, subtype=subtype)
            sizes.append(len(buf.getvalue()))
        write_secs = (time.perf_counter() - t0) / iters

        # --- read ---
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = sf.read(io.BytesIO(buf.getvalue()), dtype="float32")
        read_secs = (time.perf_counter() - t0) / iters

        rows.append(
            (
                subtype,
                write_secs,
                read_secs,
                np.mean(sizes) / 1024.0,  # KiB
            )
        )
    return rows


# ────────────────────────── pretty printing ───────────────────────────
def pretty(rows: List[Row]) -> None:
    try:
        from tabulate import tabulate

        print(
            tabulate(
                [
                    (fmt, f"{w:7.4f}", f"{r:7.4f}", f"{kib:7.1f}")
                    for fmt, w, r, kib in rows
                ],
                headers=["fmt", "write s", "read s", "KiB"],
                tablefmt="github",
            )
        )
    except ImportError:
        hdr = ["fmt", "write s", "read s", "KiB"]
        colw = [
            max(
                len(h),
                *(len(f"{v:7.4f}") if isinstance(v, float) else len(v) for v in col),
            )
            for h, col in zip(hdr, zip(*rows))
        ]
        line = "  ".join(h.ljust(w) for h, w in zip(hdr, colw))
        print(line)
        print("-" * len(line))
        for fmt, w, r, kib in rows:
            print(
                f"{fmt.ljust(colw[0])}  "
                f"{w:7.4f}  {r:7.4f}  {kib:7.1f}".ljust(sum(colw) + 6)
            )


# ─────────────────────────────── main ─────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="In-memory Ogg (Vorbis/Opus) benchmark"
    )
    parser.add_argument(
        "-n", "--iters", type=int, default=5, help="iterations per codec (default 5)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pretty(bench(args.iters))
