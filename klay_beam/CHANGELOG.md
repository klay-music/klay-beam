# v0.13.4
- Minor fixes to GitHub workflow for publishing to Docker Hub.
- Also update default `apache_beam` version to 2.53 in Dockerfile.conda

# v0.13.3
- `INCLUDE_KLAY_DATA` variable can now be used to build a docker container `klay-data` included
- New environments: `py3.10-torch2.1-klay_data3.0`, `py3.10-torch2.1-cuda12.1`, `py3.9-torch2.1-cuda12.1`

# v0.13.2

- Set default `apache_beam` in CI version to `2.53.0`
- Add GitHub workflow for publishing to PyPI

# v0.13.1

- Add tests for `numpy_to_wav`, `numpy_to_mp3` and `numpy_to_ogg`.
- Ensure `py.typed` is present in pip package
- Add `overwrite=False` parameter to SkipCompleted (#73)
- Run tests in docker container during build
- Add two build configurations with apache_beam 2.53.0. The next release may
  default to `2.53.0`
  - `PY_VERSION=3.10; BEAM_VERSION=2.53.0;`
  - `PY_VERSION=3.10; BEAM_VERSION=2.53.0; TORCH_VERSION=2.0;`

# v0.13.0

- `LoadWithLibrosa` properly handles invalid audio files, matching behavior of `LoadWithTorachaudio`
- Remove `klay_beam.transforms.MultiMatchFiles` in favor of native transforms
  - `readable_files = p | MultiMatchFiles(["**.wav", "**.mp3"])` (obsolete)
  - `readable_files = p | beam.Create(["**.wav", "**.mp3"]) | beam.io.fileio.MatchAll()`  (use this instead)

# v0.12.3

- Add `klay_beam.path.remove_suffix_pattern` with tests
- Add deprecation warning to `klay_beam.transforms.MultiMatchFiles`. This should
  be replaced with `p | beam.Create(patterns) | MatchAll()`. See
  [35e418](https://github.com/klay-music/klay-beam/commit/35e4184cb549cd8533e548733e7a6d9df9d35348)
  for details.

# v0.12.2

- Add run_cuda_test example

# v0.12.1

- Add type annotations to `write_file` and `extract_wds_id_and_ext` transform functions

# v0.12.0

- Remove Transform for extracting chroma features

# v0.11.0

- First version hosted on PYPI
