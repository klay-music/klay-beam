import apache_beam as beam
from essentia.standard import (
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
    TempoCNN,
    TensorflowPredictVGGish,
)
import logging
import numpy as np
from pathlib import Path
import torch
import traceback
from typing import Optional, Tuple

from klay_beam.path import remove_suffix


MODEL_DIR = Path.cwd() / "models"
ALL_FEATURES = [
    "genre_discogs400",
    "mtg_jamendo_genre",
    "approachability",
    "danceability",
    "engagement",
    "mood_aggressive",
    "mood_happy",
    "mood_party",
    "mood_relaxed",
    "mood_sad",
    "mtg_jamendo_moodtheme",
    "mtg_jamendo_instrument",
    "mood_acoustic",
    "mood_electronic",
    "voice_instrumental",
    "timbre",
    "nsynth_instrument",
    "nsynth_reverb",
    "tonal_atonal",
    "mtg_jamendo_top50tags",
    "mtt",
    "audioset_yamnet",
]
ESSENTIA_CLASSIFIERS = {
    "genre_discogs400": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "genre_discogs400-discogs-effnet-1.pb"),
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    ),
    "mtg_jamendo_genre": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mtg_jamendo_genre-discogs-effnet-1.pb")
    ),
    "approachability": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "approachability_2c-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "danceability": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "danceability-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "engagement": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "engagement_2c-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "mood_aggressive": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mood_aggressive-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "mood_happy": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mood_happy-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "mood_party": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mood_party-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "mood_relaxed": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mood_relaxed-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "mood_sad": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mood_sad-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "mtg_jamendo_moodtheme": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mtg_jamendo_moodtheme-discogs-effnet-1.pb")
    ),
    "mtg_jamendo_instrument": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mtg_jamendo_instrument-discogs-effnet-1.pb")
    ),
    "mood_acoustic": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mood_acoustic-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "mood_electronic": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mood_electronic-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "voice_instrumental": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "voice_instrumental-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "timbre": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "timbre-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "nsynth_instrument": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "nsynth_instrument-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "nsynth_reverb": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "nsynth_reverb-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "tonal_atonal": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "tonal_atonal-discogs-effnet-1.pb"),
        output="model/Softmax",
    ),
    "mtg_jamendo_top50tags": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mtg_jamendo_top50tags-discogs-effnet-1.pb")
    ),
    "mtt": TensorflowPredict2D(
        graphFilename=str(MODEL_DIR / "mtt-discogs-effnet-1.pb")
    ),
    "audioset_yamnet": TensorflowPredictVGGish(
        graphFilename=str(MODEL_DIR / "audioset-yamnet-1.pb"),
        input="melspectrogram",
        output="activations",
    ),
}
assert sorted(ALL_FEATURES) == sorted(ESSENTIA_CLASSIFIERS.keys()), (
    "All features must have a classifier in ESSENTIA_CLASSIFIERS."
)
DISCOGS_EFFNET_CLASSIFIERS = [
    f for f in ESSENTIA_CLASSIFIERS.keys() if f != "audioset_yamnet"
]


class ExtractEssentiaFeatures(beam.DoFn):
    embed_model_filename = "discogs-effnet-bs64-1.pb"

    def __init__(self, audio_suffix: str, features: Optional[list[str]] = None):
        self.audio_suffix = audio_suffix
        self.features = features or ALL_FEATURES

    def setup(self):
        self.model = TensorflowPredictEffnetDiscogs(
            graphFilename=str(MODEL_DIR / self.embed_model_filename),
            output="PartitionedCall:1",
        )
        self.classifiers = {
            f: ESSENTIA_CLASSIFIERS[f] for f in self.features if f != "tempo"
        }

    @property
    def suffixes(self) -> dict[str, str]:
        return {f: f".{f}.npy" for f in self.features}

    def process(self, element: Tuple[str, torch.Tensor, int]):
        fname, audio, sr = element
        audio = audio.numpy()

        # validation
        assert sr == 16_000, f"DiscogsEffnet expects 16k audio. Found {sr}. ({fname})"
        if audio.ndim == 2:
            if audio.shape[0] == 1:
                audio = audio[0]
            elif audio.shape[0] == 2:
                audio = audio.mean(axis=0)
        elif audio.ndim == 1:
            pass
        else:
            raise ValueError(
                f"DiscogsEffnet expect 1D mono audio, got shape: {audio.shape}"
            )

        logging.info(f"Found audio file: {fname} with length: {len(audio)} samples.")

        # prepare file
        for feat_name in self.features:
            suffix = self.suffixes[feat_name]
            out_filename = remove_suffix(fname, self.audio_suffix)
            out_filename += suffix
            # extract
            try:
                if feat_name in DISCOGS_EFFNET_CLASSIFIERS:
                    embeds = self.model(audio)
                    classifier = self.classifiers[feat_name]
                    preds = classifier(embeds).transpose()
                    yield out_filename, preds
                else:
                    classifier = self.classifiers[feat_name]
                    preds = classifier(audio).transpose()
                    yield out_filename, preds
            except Exception:
                logging.error(traceback.format_exc())
                return []


class ExtractEssentiaTempo(beam.DoFn):
    def __init__(self, audio_suffix: str):
        self.audio_suffix = audio_suffix

    def setup(self):
        self.model = TempoCNN(graphFilename=str(MODEL_DIR / "deeptemp-k16-3.pb"))

    def process(self, element: Tuple[str, np.ndarray, int]):
        fname, audio, sr = element
        audio = audio.numpy()

        # validation
        assert sr == 11250, f"Tempo expects 11.25k audio. Found {sr}. ({fname})"
        if audio.ndim == 2:
            if audio.shape[0] == 1:
                audio = audio[0]
            elif audio.shape[0] == 2:
                audio = audio.mean(axis=0)
        elif audio.ndim == 1:
            pass
        else:
            raise ValueError(f"Tempo expect 1D mono audio, got shape: {audio.shape}")

        logging.info(f"Found audio file: {fname} with length: {len(audio)} samples.")

        # prepare file
        out_filename = remove_suffix(fname, self.audio_suffix) + ".tempo.npy"

        # extract
        try:
            global_tempo, _, _ = self.model(audio)
            yield out_filename, np.array([[global_tempo]])
        except Exception:
            logging.error(traceback.format_exc())
            return []
