from pathlib import PurePosixPath
from urllib.parse import urlparse
import os.path
import platform

"""
When processing datasets, we often have multi-layer directories, and we need to
map input files to output files. For example, inputs could like this:

```
gs://klay-datasets/mtg_jamendo_autotagging/audios/00/1009600.mp3
gs://klay-datasets/mtg_jamendo_autotagging/audios/00/1012000.mp3
gs://klay-datasets/mtg_jamendo_autotagging/audios/01/1009601.mp3
gs://klay-datasets/mtg_jamendo_autotagging/audios/01/1009701.mp3
```

And we want to transform them to outputs like this:
```
gs://some-other-bucket/jamendo/00/1009600.something.wav
gs://some-other-bucket/jamendo/00/1012000.something.wav
gs://some-other-bucket/jamendo/01/1009601.something.wav
gs://some-other-bucket/jamendo/01/1009701.something.wav
```

This transformation is fully specified by 
- the input root path: `gs://klay-datasets/mtg_jamendo_autotagging/audios/`
- the glob pattern: `**/*.mp3`
- a lambda function that 
   - accepts a relative path like `00/1009600.mp3`
   - returns a full output path like 
     `gs://some-other-bucket/jamendo/00/1009600.something.wav`
"""
def get_target_path(input_uri: str, source_dir:str, target_dir: str) -> str:
    relative_source_filename = PurePosixPath(input_uri).relative_to(source_dir)
    relative_target_filename = relative_source_filename.with_suffix(".source.wav")

    # pathlib does not properly handle `//` in URIs
    # `assert str(pathlib.Path("gs://data/")) == "gs://data/"` fails
    #
    # As a result, we just use os.path.join to join the target_dir and the
    return os.path.join(target_dir, relative_target_filename)

# os.path.join (used in get_target_path will break on windows
assert platform.system() != "Windows"


