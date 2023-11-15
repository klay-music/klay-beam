import numpy as np

from job_rms.transforms import ExtractRMSFeatures


def test_feat_suffix():
    extractor = ExtractRMSFeatures()
    assert extractor.frame_rate == 50
    assert extractor.feat_suffix == ".rms_50hz.npy"


def test_hop_length():
    extractor = ExtractRMSFeatures()
    assert extractor.audio_sr == 16_000
    assert extractor.frame_rate == 50
    assert extractor.hop_length == 320


def test_extract_rms():
    extractor = ExtractRMSFeatures()
    audio = np.zeros(16_000)
    rms = extractor.extract_rms(audio, 320, 1280)
    assert rms.shape == (1, 51)
