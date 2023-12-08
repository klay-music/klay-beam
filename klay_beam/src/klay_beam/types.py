from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np


"""Basic klay_beam io types used when reading and writing data.

Why are these types frozen? This is to avoid accidentally modifying the data. In
distributed runners, the types passed between PTransforms should be immutable.
See: https://beam.apache.org/documentation/programming-guide/#immutability
"""


@dataclass(frozen=True)
class Datum:
    """Base type for data that is read and written via klay_beam transforms.
    
    Why are these properties optional? At different stages of a pipeline,
    different properties will be available. For example LoadWithLibrosa will
    populate the source_path, but not the target_path. A downstream transform
    that creates multiple files may not populate source_path for each file.

    Using a single reusable type for the majority of klay_beam transforms keeps
    things simple, but also means that Transform authors should be careful to
    assert that the properties they need are populated.

    What is the difference between source_PATH and source_DIR? Paths for for
    full file names. Dirs are the root of a copy operation. We've found in the
    past the past it is helpful to be able to write transforms that COPY data
    from one path to another while recursively preserving the directory
    structure from in some source_dir.
    """

    source_dir: Optional[str] = None
    target_dir: Optional[str] = None
    source_path: Optional[str] = None
    target_path: Optional[str] = None

    binary_data: Optional[bytes] = None
    """When a klay_beam pipeline is ready to WRITE a file, it should populate
    `binary_data` AND `target_path`. Note that transforms that LOAD time-series
    data do not need to populate `binary_data`. Instead transforms that load
    data will generally just populate the .datum property of a NumpyDatum or
    TorchDatum instance, and leave `binary_data` empty."""


@dataclass(frozen=True)
class NumpyDatum(Datum):
    """General data storage returned by transforms that load time-series numpy
    data such as LoadWithLibrosa"""

    datum: Tuple[np.ndarray, int]


@dataclass(frozen=True)
class TorchDatum(Datum):
    """General data storage returned by transforms that load time-series torch
    data such as LoadWithPytorch"""

    datum: Tuple["torch.Tensor", int]


@dataclass(frozen=True)
class NumpyData():
    """General data storage returned by transforms that do not have a one-to-one
    mapping between input and output files"""

    data: List[NumpyDatum] = []


@dataclass(frozen=True)
class TorchData():
    """General data storage returned by transforms that do not have a one-to-one
    mapping between input and output files"""

    data: List[TorchDatum] = []
