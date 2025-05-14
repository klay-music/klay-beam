import soundfile as sf
import numpy as np
import pathlib


if __name__ == "__main__":
    buf = np.random.randn(2, 48_000).astype("float32")  # 1 s stereo
    codecs = {
        "wav": {},
        "ogg": {"subtype": "VORBIS"},
        "opus": {"format": "ogg", "subtype": "OPUS"},
        "flac": {},
        "mp3": {},
    }
    root = pathlib.Path("/tmp/audio_test")
    root.mkdir(exist_ok=True)
    for ext, kw in codecs.items():
        f = root / f"test.{ext}"
        try:
            sf.write(f, buf.T, 48_000, **kw)
        except Exception as e:
            print(f"{ext.upper():4s}  write ✗  ({e})")
            continue
        try:
            data, sr = sf.read(f)
            print(f"{ext.upper():4s}  write ✓  read ✓  ({data.shape}, {sr} Hz)")
        except Exception as e:
            print(f"{ext.upper():4s}  read  ✗  ({e})")
