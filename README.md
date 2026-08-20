# Auto-Labeling Endoscopic Instruments, SAM-Based Segmentation Pipeline

A two-stage **Instrument Prediction Pipeline** that automatically detects and
segments surgical instruments in laparoscopic videos/images, and writes the
results out as COCO-format annotations — built to support training data
generation for surgical AI / computer vision research.

**Stage 1 — GroundingDINO**: open-vocabulary detection finds candidate
regions of interest (ROIs) containing instruments, from a free-text prompt.
**Stage 2 — SAM (Segment Anything Model)**: refines each ROI into a precise
pixel-level segmentation mask.

Orchestration is handled by **Apache Airflow**; inference is served by a
**FastAPI** REST service. See [`docs/architecture.md`](docs/architecture.md)
for the full design rationale and data-flow diagram.

---

## 1. Project structure

```
endoscopic-instrument-pipeline/
├── app/                          # FastAPI inference service
│   ├── main.py                   #   app entrypoint, model loading at startup
│   ├── config.py                 #   environment-driven settings
│   ├── schemas.py                #   request/response models
│   ├── models/
│   │   ├── grounding_dino_wrapper.py   # Stage 1: ROI detection
│   │   ├── sam_wrapper.py              # Stage 2: mask refinement
│   │   └── pipeline.py                 # ties both stages together
│   ├── routers/
│   │   ├── predict.py            #   POST /api/v1/predict, /predict/batch
│   │   └── health.py             #   GET  /api/v1/health
│   └── utils/
│       ├── image_utils.py        #   Base64 <-> PIL helpers
│       └── coco_formatter.py     #   mask -> COCO polygon/bbox/area
│
├── airflow/
│   ├── dags/
│   │   └── instrument_labeling_dag.py   # the orchestration DAG
│   └── plugins/instrument_pipeline/
│       ├── frame_extractor.py    #   video -> frames (OpenCV, on-the-fly)
│       ├── api_client.py         #   calls the FastAPI service, with retries
│       └── coco_writer.py        #   merges per-batch results, writes output
│
├── scripts/
│   ├── run_local_pipeline.py     # run the pipeline on one video, no Airflow
│   ├── prewarm_models.py         # pre-download model checkpoints
│   └── setup_env.sh              # local (non-Docker) venv setup
│
├── tests/                        # unit tests (pytest), no GPU required
├── config/pipeline_config.yaml   # reference parameter table
├── data/                         # input_videos/, input_images/, output/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.airflow
├── requirements/{api.txt, airflow.txt}
├── .env.example
└── docs/architecture.md
```

---

## 2. Prerequisites

- Docker and Docker Compose v2
- (Recommended) an NVIDIA GPU + the NVIDIA Container Toolkit, for practical
  inference speed. The pipeline runs on CPU too (`DEVICE=cpu`), just slower.
- Internet access on first run, to download model checkpoints from Hugging
  Face Hub (~170 MB for GroundingDINO-tiny, ~375 MB for SAM-vit-base).

---

## 3. Quick start (Docker, full stack)

```bash
# 1. Configure environment
cp .env.example .env
# edit .env if you want to change models, thresholds, or the Airflow admin password

# 2. Build and start everything (Postgres, Airflow, FastAPI inference service)
docker compose up -d --build

# 3. (Optional but recommended) pre-download model checkpoints so the first
#    real DAG run isn't slowed down by the download:
docker compose exec instrument-api python scripts/prewarm_models.py

# 4. Confirm the inference API is healthy
curl http://localhost:8000/api/v1/health
# {"status":"ok","device":"cuda","grounding_dino_loaded":true,"sam_loaded":true}

# 5. Open the Airflow UI
#    http://localhost:8080   (login: admin / admin, or your .env values)
```

### Run the pipeline

1. Drop your video files into `data/input_videos/` (`.mp4`, `.avi`, `.mov`, `.mkv`).
2. In the Airflow UI, un-pause and trigger the **`instrument_labeling_pipeline`** DAG
   (or via CLI: `docker compose exec airflow-scheduler airflow dags trigger instrument_labeling_pipeline`).
3. Results land in `data/output/`:
   - `<video_name>_annotations.json` — COCO-format annotations for that video
   - `run_manifest_<timestamp>.json` — summary across all videos in that run

### DAG parameters (overridable per-run from the Airflow UI's "Trigger DAG w/ config")

| Param | Default | Meaning |
|---|---|---|
| `frame_interval` | `5` | Keep 1 out of every N frames |
| `batch_size` | `8` | Frames per inference API call |
| `text_prompt` | `"surgical instrument. forceps. scissors. grasper. clip applier. needle holder."` | Open-vocabulary prompt for GroundingDINO |
| `input_video_dir` | `/opt/airflow/data/input_videos` | Where the DAG looks for videos |
| `output_dir` | `/opt/airflow/data/output` | Where annotations are written |

---

## 4. Quick start (no Docker — single video, local Python)

Useful for fast iteration on model thresholds before wiring up Airflow.

```bash
bash scripts/setup_env.sh
source .venv/bin/activate

python scripts/run_local_pipeline.py \
    --video data/input_videos/your_clip.mp4 \
    --output-dir data/output \
    --frame-interval 5
```

This imports the pipeline directly (no HTTP call, no Airflow) and writes the
same `<video_name>_annotations.json` COCO file.

---

## 5. Calling the API directly

```bash
# Single frame
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$(base64 -w0 your_frame.jpg)\", \"file_name\": \"frame_0001.jpg\"}"
```

Interactive Swagger docs are available at `http://localhost:8000/docs` once
the service is running.

### Example response (COCO format)

```json
{
  "images": [{"id": 1, "file_name": "frame_0001.jpg", "width": 1920, "height": 1080}],
  "annotations": [
    {
      "id": 1, "image_id": 1, "category_id": 1, "category_name": "forceps",
      "bbox": [812.3, 401.7, 220.5, 95.2],
      "area": 14380.0,
      "segmentation": [[812.3, 401.7, 1032.8, 401.7, "..."]],
      "iscrowd": 0, "score": 0.91
    }
  ],
  "categories": [{"id": 1, "name": "forceps", "supercategory": "instrument"}],
  "instrument_count": 1
}
```

---

## 6. Running the tests

```bash
bash scripts/setup_env.sh        # if not already done
source .venv/bin/activate
pip install pytest pytest-cov httpx
pytest
```

Tests cover the COCO formatter, image encode/decode, frame extraction (using
a synthetic generated video, no sample footage required), and COCO batch
merging — all logic that's correct-or-broken independent of which model
checkpoint is loaded, so they run on CPU with no GPU and no downloaded
weights required.

---

## 7. Configuration reference

All runtime configuration is environment-variable driven (`app/config.py`),
documented in [`.env.example`](.env.example). Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `GROUNDING_DINO_MODEL_ID` | `IDEA-Research/grounding-dino-tiny` | Stage 1 checkpoint |
| `SAM_MODEL_ID` | `facebook/sam-vit-base` | Stage 2 checkpoint |
| `BOX_THRESHOLD` | `0.30` | GroundingDINO confidence cutoff |
| `TEXT_THRESHOLD` | `0.25` | GroundingDINO text-match cutoff |
| `SEGMENTATION_FORMAT` | `polygon` | `polygon` or `rle` |
| `MAX_BATCH_SIZE` | `8` | Max frames per `/predict/batch` call |

---

## 8. Swapping in your own dataset

This deliverable is structured around **your own videos/images**, dropped
into `data/input_videos/` or `data/input_images/`. No dataset-specific code
changes are needed — GroundingDINO's open-vocabulary prompt
(`text_prompt`) is the only thing you may want to tune per-dataset (e.g. if
your footage uses different instrument naming conventions, or you want to
restrict detection to a narrower instrument set).

If your videos have unusual codecs OpenCV can't decode, transcode first with
`ffmpeg -i input.xyz -c:v libx264 output.mp4`.

---

## 9. GPU vs CPU

By default `Dockerfile.api` builds on an NVIDIA CUDA base image and
`docker-compose.yml` has a commented-out GPU `deploy.reservations` block.

- **On a GPU host**: uncomment that block in `docker-compose.yml` and ensure
  the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  is installed on the host.
- **On a CPU-only machine**: set `DEVICE=cpu` in `.env`, and optionally swap
  `Dockerfile.api`'s base image from `nvidia/cuda:...` to `python:3.11-slim`
  to avoid pulling the CUDA runtime unnecessarily (comment in the Dockerfile
  notes this).

---

## 10. What was already verified in this environment

Every Python file in this project was syntax-checked, and the logic that
doesn't require GPU/model downloads (COCO formatting, image encode/decode,
video frame extraction against a generated synthetic video, COCO batch
merging) was executed directly and passed. The model-dependent paths
(GroundingDINO/SAM inference, the FastAPI app, the live Airflow DAG) require
`torch`/`transformers`/`fastapi`/`apache-airflow` plus a model-weight
download, none of which are available in the sandboxed environment this was
authored in — these should be exercised in your Docker/GPU environment per
the Quick Start above. The full `pytest` suite in `tests/` is ready to run
there too.
