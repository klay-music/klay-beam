from __future__ import annotations

import concurrent.futures as cf
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import random
from google.cloud import storage
from tqdm import tqdm

from job_stats.genre_classes import GENRE_CLASSES
from klay_data.feature_loader import AudiosetYamnetFeatureLoader


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GCS_DATASET_ROOT: str = "gs://klay-datasets-test"  # bucket URI without trailing slash
METADATA_JSON_PATH: Path = Path(OUTPUT_DIR / "metadata.json")
FULL_METADATA_JSON_PATH: Path = Path(OUTPUT_DIR / "full_metadata.json")
PLOTS_DIR: Path = Path(OUTPUT_DIR / "plots")
SAMPLE_ARCHIVE_PATH: Path = Path(OUTPUT_DIR / "audio_samples")

GENRE_THRESHOLD: float = 0.8  # probability threshold for genre assignment
NUM_TOP_GENRES: int = 10
NUM_SAMPLES_PER_GENRE: int = 10
NUM_SAMPLES = 50_000
MAX_WORKERS: int | None = os.cpu_count()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    assert gcs_uri.startswith("gs://"), f"Invalid GCS URI: {gcs_uri}"
    bucket_name, _, blob_path = gcs_uri[5:].partition("/")
    return bucket_name, blob_path


def list_track_subdirectories(bucket_name: str, client: storage.Client) -> List[str]:
    subdirs: set[str] = set()

    shards_iter = client.list_blobs(bucket_name, prefix="shard-", delimiter="/")
    for shards_page in shards_iter.pages:
        for shard_prefix in shards_page.prefixes:
            tracks_iter = client.list_blobs(
                bucket_name, prefix=shard_prefix, delimiter="/"
            )
            for tracks_page in tracks_iter.pages:
                for track_prefix in tracks_page.prefixes:
                    subdirs.add(f"gs://{bucket_name}/{track_prefix}")
    return sorted(subdirs)


def download_bytes(gcs_uri: str, client: storage.Client) -> bytes:
    bucket_name, blob_path = parse_gcs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes()


class VocalDetector:
    p_threshold: float
    n_threshold: float

    def __init__(self, p_threshold: float = 0.9, n_threshold: float = 0.1):
        self.p_threshold = p_threshold
        self.n_threshold = n_threshold

    @staticmethod
    def majority_vote(
        probs: np.ndarray, p_threshold: float = 0.9, n_threshold: float = 0.1
    ) -> bool:
        is_vocal_frames = probs[0] > p_threshold
        total_is_vocal = np.sum(is_vocal_frames) / probs.shape[1]
        return total_is_vocal >= n_threshold

    def __call__(self, classifier_array: np.ndarray) -> bool:
        return self.majority_vote(classifier_array, self.p_threshold, self.n_threshold)


def assign_genres(
    genre_logits: np.ndarray, threshold: float = GENRE_THRESHOLD
) -> List[str]:
    genre_idx = genre_logits.max(axis=-1).argmax()
    return [GENRE_CLASSES[genre_idx]]


def compute_is_music(yamnet_features: np.ndarray) -> bool:
    loader_cls = AudiosetYamnetFeatureLoader
    top_label, top_logit = loader_cls._get_top_label(yamnet_features)
    return (
        top_label == loader_cls.music_class_label
        and top_logit >= loader_cls.music_prob_threshold
    )


def read_duration_seconds(audio_bytes: bytes) -> float:
    with sf.SoundFile(io.BytesIO(audio_bytes)) as snd:
        frames = len(snd)
        sr = snd.samplerate
    return frames / sr


def process_single_track(subdir_uri: str) -> Dict[str, Any]:
    client = storage.Client()
    bucket_name, prefix_path = parse_gcs_uri(subdir_uri)
    track_id = Path(prefix_path.rstrip("/")).name

    def blob_uri(suffix: str) -> str:
        return f"gs://{bucket_name}/{prefix_path}{track_id}{suffix}"

    yamnet_uri = blob_uri(".source.audioset_yamnet.npy")
    voice_instr_uri = blob_uri(".source.voice_instrumental.npy")
    genre_uri = blob_uri(".source.genre_discogs400.npy")
    ogg_source_uri = blob_uri(".source.ogg")

    try:
        yamnet_features = np.load(io.BytesIO(download_bytes(yamnet_uri, client)))
        voice_instr_arr = np.load(io.BytesIO(download_bytes(voice_instr_uri, client)))
        genre_logits = np.load(io.BytesIO(download_bytes(genre_uri, client)))
    except Exception as exc:
        logging.error(f"Missing or corrupt feature file for {track_id}: {exc}")
        return {}

    try:
        audio_bytes = download_bytes(ogg_source_uri, client)
        duration_seconds = read_duration_seconds(audio_bytes)
    except Exception as exc:
        logging.error(f"Failed to read audio for {track_id}: {exc}")
        duration_seconds = -1.0

    is_music = compute_is_music(yamnet_features)
    is_vocal = VocalDetector()(voice_instr_arr)
    genres = assign_genres(genre_logits)

    if genres:
        super_genre = [g.split("---")[0] for g in genres][0]
    else:
        super_genre = []

    return {
        "track_id": track_id,
        "subdir": subdir_uri,
        "duration_seconds": duration_seconds,
        "is_music": is_music,
        "is_vocal": is_vocal,
        "genres": genres,
        "super_genre": super_genre,
    }


def gather_dataset_statistics(track_subdirs: Sequence[str]) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for output in tqdm(
            executor.map(process_single_track, track_subdirs), total=len(track_subdirs)
        ):
            if output:
                results.append(output)
    return pd.DataFrame(results)


def save_metadata(df: pd.DataFrame, output_path: Path) -> None:
    df.to_json(output_path, orient="records", lines=True)


def plot_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    df["duration_seconds"].hist(bins=100)
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.title("Track Duration Distribution")
    plt.savefig(output_dir / "duration_histogram.png")
    plt.close()

    genre_counts: pd.Series = (
        df.explode("super_genre")["super_genre"].value_counts().head(20)
    )
    plt.figure()
    genre_counts.plot(kind="bar")
    plt.ylabel("Track Count")
    plt.title("Top 20 Genres")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "genre_distribution.png")
    plt.close()

    for column, title in [
        ("is_vocal", "Vocal Presence"),
        ("is_music", "Music Presence"),
    ]:
        plt.figure()
        df[column].value_counts().plot(
            kind="pie", autopct="%1.1f%%", labels=["True", "False"], startangle=90
        )
        plt.title(title)
        plt.ylabel("")
        plt.savefig(output_dir / f"{column}.png")
        plt.close()


def sample_tracks_by_genre(df: pd.DataFrame, output_dir: Path) -> list:
    client = storage.Client()

    top_genres: List[str] = (
        df.explode("super_genre")["super_genre"]
        .value_counts()
        .head(NUM_TOP_GENRES)
        .index.tolist()
    )
    subdir_list = []

    for genre in top_genres:
        genre_rows = df[df["super_genre"].apply(lambda g: genre in g)]
        if genre_rows.empty:
            continue
        selected_rows = genre_rows.sample(
            n=min(NUM_SAMPLES_PER_GENRE, len(genre_rows)), random_state=42
        )
        for _, row in selected_rows.iterrows():
            subdir_uri: str = row["subdir"]
            subdir_list.append(subdir_uri)
            bucket_name, prefix_path = parse_gcs_uri(subdir_uri)
            track_id: str = row["track_id"]

            def blob_uri(suffix: str) -> str:
                return f"gs://{bucket_name}/{prefix_path}{track_id}{suffix}"

            for suffix in [
                ".instrumental.ogg",
                ".vocals.ogg",
                ".source.ogg",
                ".source.webm",
                ".vocals.whisper.json",
                ".vocals.whisperx-v2.json",
            ]:
                file_uri = blob_uri(suffix)
                try:
                    file_bytes = download_bytes(file_uri, client)
                except Exception:
                    logging.warning(f"File missing: {file_uri}")
                    continue

                # write to disk
                filepath = output_dir / f"{genre}/{track_id}{suffix}"
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, "wb") as f:
                    f.write(file_bytes)

    return subdir_list


def main() -> None:
    logging.info("Initialising GCS client…")
    client = storage.Client()

    bucket_name, _ = parse_gcs_uri(GCS_DATASET_ROOT)
    logging.info("Listing track subdirectories…")
    subdirs = list_track_subdirectories(bucket_name, client)
    logging.info(f"Found {len(subdirs)} track subdirectories")

    random.shuffle(subdirs)
    subdirs = subdirs[:NUM_SAMPLES]

    logging.info("Processing tracks in parallel…")
    metadata_df = gather_dataset_statistics(subdirs)
    logging.info(f"Collected metadata for {len(metadata_df)} tracks")

    logging.info("Saving metadata to JSON…")
    save_metadata(metadata_df, FULL_METADATA_JSON_PATH)

    logging.info("Plotting dataset statistics…")
    plot_statistics(metadata_df, PLOTS_DIR)

    logging.info("Creating sample archive…")
    subdir_list = sample_tracks_by_genre(metadata_df, SAMPLE_ARCHIVE_PATH)

    # filter metadata.json by subdir_list
    metadata_df = metadata_df[metadata_df["subdir"].isin(subdir_list)]
    logging.info(f"Filtered metadata for {len(metadata_df)} tracks in sample archive")
    logging.info("Saving filtered metadata to JSON…")
    save_metadata(metadata_df, METADATA_JSON_PATH)

    logging.info("Done.")


if __name__ == "__main__":
    main()
