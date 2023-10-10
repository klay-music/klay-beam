import apache_beam as beam
from pathlib import Path
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from apache_beam.io import fileio

from klay_beam.transforms import MultiMatchFiles



def test_MultiMatchFiles():

    this_file = Path(__file__)
    data_dir = this_file.parent / "test_data"
    patterns = [str(data_dir / fn) for fn in ['00/*.txt', '01/*.txt']]
    expected = [str(data_dir / fn) for fn in ['00/a.txt', '00/b.txt', '01/a.txt', '01/b.txt']]

    with TestPipeline() as p:

        # Apply the custom transform
        results = p | MultiMatchFiles(patterns)

        # Extract file names from the metadata for assertion
        file_names = results | 'Extract File Names' >> beam.Map(lambda metadata: metadata.path)

        # Check if the results match the expected file names
        assert_that(file_names, equal_to(expected))
