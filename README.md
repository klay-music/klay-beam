# `klay_beam` – Engineering README


## 📜 Preface

This README is the **canonical, audit‑ready** introduction to the Klay Beam platform. It is intentionally verbose so that:

* **New engineers** ramp up without spelunking in Slack threads.
* **Auditors** can validate data‑flow guarantees and dependency hygiene.
* **Ops** can debug production pipelines at 03:00 without paging devs.

> If you add a feature or change a process, **update this file in the same PR** so we never drift. Compliance depends on it.


## ️🎛️ Overview

Klay Beam is our in‑house framework for *massively* parallel audio processing. It marries three layers:

| Layer            | Responsibility                                                                                                                            |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Apache Beam**  | Declarative DAG that scales from a laptop (`DirectRunner`) to thousands of machines (`DataflowRunner`).                                   |
| **Docker**       | Hermetic runtimes for local tests *and* remote workers; the identical image‑SHA runs everywhere, eliminating “works‑on‑my‑machine” drift. |
| **GCP Dataflow** | Spot‑provisioned Compute Engine VMs that pull the Beam container, hydrate it, and execute the pipeline shards in parallel.                |

> **Why Beam?** One SDK targets local threads, an on‑prem Mesos cluster, *or* fully‑managed Dataflow—no rewrite required.

### ️🏁 Data lifecycle (10 000‑ft)

1. **Ingest** – `MatchFiles` emits `FileMetadata` for every object that matches a glob.
2. **Transform** – loaders decode audio → tensors; `PTransform`s perform DSP, ML inference, feature extraction.
3. **Persist** – results land in Cloud Storage / BigQuery / Firestore depending on the pipeline.
4. **Observe** – Cloud Logging, Profiler, Error Reporting capture metrics & failures.


## 🧩 Repository Anatomy

```
.
├── README.md          ← This document
├── klay_beam/         ← Core Python package *+* base Docker image
└── jobs/              ← One sub‑dir per job
    ├── job_whisper/   ← GPU‑heavy Whisper speech‑to‑text
    ├── job_byt5/      ← CPU‑only BYT5 tagging
    └── …
```

*Every* path in `jobs/**` is an **independent** Python package. A job *may* add a `Dockerfile` that layers on top of the **base image** shipped in `klay_beam`—but only when truly required (see below).


## 🏗️ Jobs

### What *is* a job?

A **job** is a self‑contained Beam pipeline with a single business goal (e.g. stem separation, lyrics extraction, embedding generation). Each lives under `jobs/job_<name>/` and must provide:

| File/Dir                                | Purpose                                                                                                  |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `Dockerfile`                            | Adds *only* the deltas (models, extra wheels, GPU drivers) on top of the base image.                     |
| `Makefile`                              | Canonical interface for humans *and* CI. Targets:                                                        |
| • `code-style` · `type-check` · `tests` |                                                                                                          |
| • `docker` · `docker-push`              |                                                                                                          |
| • `run-local` · `run-dataflow`          |                                                                                                          |
| `bin/`                                  | Thin CLI wrappers (usually `run_workflow.py`) that assemble `PipelineOptions` and launch the Beam graph. |
| `environment/`                          | *Optional* **conda‑lock** for heavyweight native deps that cannot be `pip`‑installed.                    |
| `src/`                                  | Package code (`import job_whisper …`).                                                                   |
| `tests/`                                | PyTest suite; required for merge (≥ 80 % new branch coverage).                                           |
| `README.md`                             | Short operational doc (mirrors Makefile flags).                                                          |

> **Philosophy** – Extend the base image **only when** you need a private model, conflicting dependency versions, or GPU‑specific binaries. Otherwise inherit directly; containers stay small 🏃‍♀️.

### CPU vs GPU templates

| Scenario                 | Template                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ |
| CPU‑only workload        | **`jobs/job_byt5`** – lean, single‑thread performance paramount.                                             |
| GPU‑accelerated workload | **`jobs/job_whisper`** – shows how to request NVIDIA L4s and install drivers via `dataflow_service_options`. |

### Creating a new job 🚀

1. **Scaffold**

   ```bash
   cp -r jobs/job_byt5 jobs/job_mynew
   find jobs/job_mynew -type f -exec sed -i '' -e 's/job_byt5/job_mynew/g' {} +
   ```
2. **Update metadata** – `pyproject.toml` name is already patched; set version → `0.1.0` in `__init__.py`.
3. **Decide on Dockerfile**

   | Question                                    | If **yes** ..                                  |
   | ------------------------------------------- | ---------------------------------------------- |
   | Need extra *system* libs?                   | Write a Dockerfile and `apt-get` them.         |
   | Need extra *Python* deps? (pip‑installable) | Add to `pyproject.toml`; no Dockerfile needed. |
   | Need private local packages or big models?  | Write a Dockerfile and `COPY` / install.       |

   Always start `FROM ${BASE_IMAGE}` (echo it with `make print-base`).
4. **Pipeline code** – in `bin/run_workflow.py`. Leverage existing transforms in **`klay_beam.transforms`** before writing new ones. Custom transforms → `src/job_mynew/transforms.py`.
5. **Tests** – `pytest -q`; aim ≥ 80 % coverage for new code.
6. **Commit & tag**

   ```bash
   git add .
   git commit -m 'job-mynew: 0.1.0'
   git tag -a job-mynew-0.1.0 -m 'Initial release'
   git push --tags
   ```

   Tagging triggers **Cloud Build** → pushes `us-docker.pkg.dev/…/job-mynew:0.1.0`.

### Launching jobs

| Environment  | Command                                                           |
| ------------ | ----------------------------------------------------------------- |
| **Local**    | `make run-local match_pattern="~/audio/**.wav" audio_suffix=.wav` |
| **Dataflow** | `make run-dataflow job_name=mynew max_num_workers=128 \`          |

```
            `match_pattern='gs://bucket/audio/**.wav' audio_suffix=.wav` |
```

All runtime knobs (`--max_num_workers`, `--runner`, `--temp_location`, …) live in the Makefile; override them inline via `make VAR=value …`.


# 📦 `klay_beam` Core Library

* **Reusable transforms** – file IO, resampling, channel ops, `SkipCompleted`, file‑name mutation, `FeatureWriter`, etc.
  Undocumented gems worth noting:

  * **`SkipCompleted`** – checks output path and drops elements already processed; idempotent reruns cost \$0.
  * **`FileNameMutator`** – atomic `gs://bucket/{fname}` renames that survive pre‑emption.
  * **`FeatureWriter`** – writes NumPy/Parquet features with schema injection.
* **Utility CLI** – `python -m klay_beam.run_example` showcases an end‑to‑end local pipeline (decode → duration sum).
* **Scripts** – `bin/create_test_audio_files.py` generates synthetic WAV/OGG for smoke tests; `bin/test-soundfile.py` validates libsndfile linkages.
* **Base Docker image** – `klay_beam/Dockerfile.conda` is our golden runtime – every job inherits from this.
* **Dev helpers** – `docker-env-helper.sh` computes build args (`PY_VERSION`, `TORCH_VERSION`, etc.) for CI matrix.

## Local development

| Task                    | Command                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------ |
| **Create env**          | `git submodule update --init --recursive && conda env create -f environment/py3.10-torch2.5.yml` |
| **Run tests**           | `make tests`                                                                                     |
| **Lint + format**       | `make code-style` *(flake8 + black --check)*                                                     |
| **Static types**        | `make type-check` *(mypy)*                                                                       |
| **Generate conda‑lock** | `make conda-lock` whenever `environment/*.yml` changes                                           |

## Release process

1. Implement feature/bugfix → ensure tests pass.
2. Bump `src/klay_beam/__init__.py::__version__`, update `CHANGELOG.md`.
3. `git tag v$(./get_version.sh)` & push.
4. GitHub Actions publishes to **PyPI** (`publish-pypi` branch) then builds **DockerHub** images (`publish-docker-hub`).

Every published image tag encodes:
`klay_beam-${VERSION}-py${PY_VERSION}-beam${BEAM_VERSION}-torch${TORCH_VERSION}[-cuda${CUDA_VERSION}]`


# 🔧 DevOps Pipeline

## Base image (GitHub Actions)

* **Workflow** – `.github/workflows/publish-docker-hub.yaml`.
* **Generate lock** – `make conda-lock` when env file changes. CI fails if lock is out‑of‑date.
* **Matrix** – Python 3.10 × Torch {2.1, 2.5} × {CPU, CUDA 12.1} × Beam `${BEAM_VERSION}`.
* **Secrets** – SSH deploy keys for private submodules, DockerHub creds.
* **Multi‑stage build** – `Dockerfile.conda`:

  1. Pull Beam SDK image (for worker harness binaries).
  2. Build conda env with **mambaforge**.
  3. Produce slim runtime (`python:3.10-slim-bookworm`) + `RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT=1`.

## Job images (Cloud Build)

* **Trigger** – `git tag job-<name>-<semver>` → `git push --tags`.
* **Steps** – `cloudbuild.yaml` clones submodules, builds `jobs/job_<name>/Dockerfile`, pushes to Artifact Registry (`us-docker.pkg.dev/klay-home/klay-docker`).
* **Image TTL** – immutable; older tags never deleted (audit trail).

## Versioning rules

| Artifact            | How to bump                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| **klay\_beam**      | semantic version in `src/klay_beam/__init__.py` → tag `vX.Y.Z`.          |
| **Job**             | bump `src/job_<name>/__init__.py::__version__` → tag `job-<name>-X.Y.Z`. |
| **Docker env lock** | regenerate `conda-lock` → commit (hash appears in image tag).            |

## Continuous quality gates

* Push → PR runs `pytest`, `flake8`, `black --check`, `mypy`.
* `main` branch is protected; must stay **green**.


# 🧠 Troubleshooting

1. **Reproduce locally** with the exact container:

   ```bash
   ```

docker run -it --entrypoint /bin/bash gcr.io/<region>/klay-beam:<tag>

```
2. **Inspect logs** – Cloud Logging, filter `labels.dataflow.googleapis.com/step_id`.

## Common issues & fixes
| Symptom | Likely cause | Resolution |
| ------- | ------------ | ---------- |
| Pipeline hangs at start | Fusion across huge `MatchFiles` | Insert `beam.Reshuffle()` after glob. |
| `No CUDA devices` | Wrong `machine_type` or `worker_accelerator` | Match GPU type (`nvidia-l4`, `a100`, etc.) + quota. |
| `Permission denied` when writing temp | SA lacks `storage.objectAdmin` on temp bucket | Grant role or change `--temp_location`. |
| `ErrImagePull` | Image tag typo or not pushed yet | Verify image exists in Artifact Registry and tag matches lock hash. |
| `ModuleNotFoundError` for private lib | Dockerfile forgets to COPY submodule | Add `COPY --from=builder /submodules/…` or `pip install -e`. |
| Beam creates new venv | Missing `RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT=1` | Ensure env var in image. |

---

> Remember: **every README is a compliance artifact**. Keep it current so auditors stay happy and *nobody dies* 🤞
