from __future__ import annotations

import pytest
import numpy as np

from job_byt5.transforms import ExtractWhisperByT5


@pytest.fixture(scope="session")
def real_dofn() -> ExtractWhisperByT5:
    """
    Creates an instance of the ExtractWhisperByT5 DoFn and calls its setup()
    method with a real ByT5TextExtractor (no mocking).

    NOTE: This loads the 'google/byt5-small' model, which can be slow.
    """
    fn = ExtractWhisperByT5()
    fn.setup()  # sets up self.device and self.extractor with the real ByT5 model
    return fn


@pytest.fixture
def valid_sample_dict() -> dict:
    """
    Returns a sample dictionary emulating WhisperX-style output with 8 total words,
    2 of which are blacklisted. Thus, we expect 6 valid words after filtering.
    """
    return {
        "lyrics": [
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 1.0},
                    {"word": "Bye.", "start": 1.1, "end": 2.0},  # blacklisted
                    {"word": "World", "start": 2.1, "end": 3.0},
                    {"word": "Thank you.", "start": 3.1, "end": 4.0},  # blacklisted
                    {"word": "Is", "start": 4.1, "end": 5.0},
                    {"word": "This", "start": 5.1, "end": 6.0},
                    {"word": "Real", "start": 6.1, "end": 7.0},
                    {"word": "Life", "start": 7.1, "end": 8.0},
                ],
                "end": 8.0,
                "start": 0.0,
            }
        ],
    }


def test_invalid_suffix(real_dofn: ExtractWhisperByT5) -> None:
    """
    If file doesn't end in .whisper.json, we expect a ValueError.
    """
    with pytest.raises(ValueError, match="must end with .whisper.json"):
        _ = list(real_dofn.process(("sample_INVALID.json", {})))


def test_empty_lyrics(real_dofn: ExtractWhisperByT5) -> None:
    """
    If we have only blacklisted words, there's nothing to yield.
    """
    empty_lyrics_dict = {
        "lyrics": [
            {"word": "Bye.", "start": 0.0, "end": 1.0},  # blacklisted
            {"word": "Thank you.", "start": 1.1, "end": 2.0},  # blacklisted
        ],
        "start": 0.0,
        "end": 2.0,
    }

    # This file ends in .whisper.json, so suffix is valid
    outputs = list(
        real_dofn.process(("only_blacklisted.whisper.json", empty_lyrics_dict))
    )
    assert len(outputs) == 0, "Expected no yields when all words are blacklisted."


def test_valid_extraction(
    real_dofn: ExtractWhisperByT5, valid_sample_dict: dict
) -> None:
    """
    End-to-end test that runs the DoFn with a real ByT5TextExtractor, verifying:
      - Suffix is replaced with .whisper-byt5.npz
      - Blacklisted words are filtered
      - Embeddings/tokens/timing arrays are non-empty
      - 6 valid words remain after filtering out 2 blacklisted
    """
    # The valid file suffix is .whisper.json
    results = list(real_dofn.process(("sample.whisper.json", valid_sample_dict)))

    assert len(results) == 1, "Expected exactly one output."

    output_filepath, output_data = results[0]
    # Check the new suffix
    assert output_filepath.endswith(".whisper-byt5.npz")

    # Check output_data keys
    assert "byt5_embeds" in output_data
    assert "byt5_tokens" in output_data
    assert "starts" in output_data
    assert "ends" in output_data

    # Check that the arrays are not None or empty
    embeds = output_data["byt5_embeds"]
    tokens = output_data["byt5_tokens"]
    starts = output_data["starts"]
    ends = output_data["ends"]

    # Expect arrays to have content
    assert (
        isinstance(embeds, np.ndarray) and embeds.size > 0
    ), "Embeds should be a non-empty numpy array."
    assert (
        isinstance(tokens, np.ndarray) and tokens.size > 0
    ), "Tokens should be a non-empty numpy array."
    assert (
        isinstance(starts, np.ndarray) and starts.size > 0
    ), "Starts should be a non-empty numpy array."
    assert (
        isinstance(ends, np.ndarray) and ends.size > 0
    ), "Ends should be a non-empty numpy array."

    # Start/end arrays should match shape
    assert starts.shape == ends.shape, "Start/End arrays must match shape."

    # ByT5 small typically has embed dim ~1472 (or 1536).
    # The time dimension depends on input length
    assert embeds.shape[0] in (
        1472,
        1536,
    ), "Embedding dimension should match ByT5 small/base."
    assert embeds.shape[1] > 0, "Sequence dimension should be > 0."

    # Confirm blacklisted words ('Bye.', 'Thank you.') are absent
    # We'll decode tokens to check. (Spacing may vary.)
    decoded_word = real_dofn.extractor.tokenizer.decode(tokens)
    assert "Bye." not in decoded_word, "Blacklisted 'Bye.' should be absent."
    assert (
        "Thank you." not in decoded_word
    ), "Blacklisted 'Thank you.' should be absent."

    # Confirm that some known valid words remain
    # If 6 valid words remain, at least a few must be in the final string
    for valid_word in ("Hello", "World", "Is", "This", "Real", "Life"):
        assert (
            valid_word in decoded_word
        ), f"Expected '{valid_word}' in the decoded word, which should have 6 valid words."
