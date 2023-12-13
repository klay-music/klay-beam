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
