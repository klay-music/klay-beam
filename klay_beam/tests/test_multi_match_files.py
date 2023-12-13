import apache_beam as beam
from pathlib import Path
from apache_beam.io.fileio import MatchAll
from apache_beam.testing.test_pipeline import TestPipeline as BeamTestPipeline
from apache_beam.testing.util import assert_that, equal_to


def test_apache_beam_MatchAll():
    """Klay_beam version 0.12.x has a custom transform called MultiMatchFiles
    used to match multiple file patterns. However, this method is slow, because
    it could not parallelize.

    This test documents a better way to do this using Beam's MatchAll transform.

    See:
    https://beam.apache.org/releases/pydoc/current/apache_beam.io.fileio.html#apache_beam.io.fileio.MatchAll

    klay_beam.transforms.MultiMatchFiles should be deprecated after version
    0.12.x.
    """
    this_file = Path(__file__)
    data_dir = this_file.parent / "test_data"
    patterns = [str(data_dir / fn) for fn in ["00/*.txt", "01/*.txt"]]
    expected = [
        str(data_dir / fn) for fn in ["00/a.txt", "00/b.txt", "01/a.txt", "01/b.txt"]
    ]

    with BeamTestPipeline() as p:
        results = p | beam.Create(patterns) | MatchAll()

        # Extract file names from the metadata for assertion
        file_names = results | "Extract File Names" >> beam.Map(
            lambda metadata: metadata.path
        )

        # Check if the results match the expected file names
        assert_that(file_names, equal_to(expected))
