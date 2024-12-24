# v0.13.9
- Add `klay-data` to docker build by default
- Update to `apache_beam[gcp]==2.61.0`
- Include `boto3` in dependencies
- Add new transforms: `LoadNpy`, `LoadJson`, `MatchFiles`

# v0.13.7

- Skipped 0.13.6 due to invalid changes to `main`
- Fix bug in `ensure_torch_available` where versions comparison wasn't executed properly.
- All base docker images are now using Debian 11 for cuda-toolkit compatability.

# v0.13.5

- Skipped 0.13.3 and 0.13.4 due to invalid changes to `main`
- Minor fixes to `docker` build scripts
- New environment: `py3.10-torch2.1-cuda12.1`

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
