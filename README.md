# geo_agent_system

`geo_agent_system` is a lightweight Python framework for a human-in-the-loop, multi-agent geolocation reasoning system. It supports real OpenAI-compatible LLM/VLM calls, Serper Web search, Serper Places/Baidu Maps POI search. All model calls and paid tools require real API keys; missing keys produce a clear error or make the affected tool unavailable to the Brain.

## Goals

- Decouple Agent, Tool, Workflow, and State.
- Keep sensitive values in `.env`; keep YAML configs limited to non-sensitive behavior.
- Support interactive single-image localization and batch localization.
- Store visual cues, OCR, hypotheses, evidence, tool calls, tool results, uncertainty, and final answer in one `GeoLocalizationState`.
- Use `react` as the default LLM-driven workflow: the Brain agent decides whether to call a tool, ask the user, or finalize.
- Return the best supported location granularity, from exact coordinates down to city, region, country, continent, or unknown.

## Layout

- `configs/`: non-sensitive YAML configs and prompt templates.
- `geoagent/core/`: config loading, registries, shared schemas, logging, exceptions.
- `geoagent/state/`: unified task state.
- `geoagent/models/`: LLM/VLM clients for OpenAI-compatible endpoints.
- `geoagent/tools/`: base tool interface plus vision/search/POI/map/user tools.
- `geoagent/agents/`: perception, Brain, evidence review, memory management, and human intervention agents.
- `geoagent/memory/`: SQLite/Chroma experience storage, retrieval, reflection, and consolidation.
- `geoagent/workflows/`: ReAct workflow orchestration.
- `scripts/`: CLI entry points.
- `tests/`: automated tests.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env
```

Install the experience-memory dependency only when that subsystem is needed:

```bash
pip install ".[memory]"
```

The optional GeoCLIP evaluator additionally requires the visual dependencies and
the official `geoclip` package:

```bash
pip install ".[visual]" geoclip
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

## Configure `.env`

Fill only the values you need. All model calls and paid tools require their API keys; a missing key raises a clear error or makes the tool unavailable to the Brain. Runtime credentials are role-specific so the active model for each agent is explicit.

```dotenv
# Multimodal ReAct Brain used by the workflow and visual tools such as map verification.
BRAIN_API_KEY=
BRAIN_BASE_URL=
BRAIN_MODEL=

# Memory-writing review model. Called when memory is enabled to extract reusable
# memories from completed episodes. If omitted, the memory manager reuses BRAIN_*.
MEMORY_MANAGER_API_KEY=
MEMORY_MANAGER_BASE_URL=
MEMORY_MANAGER_MODEL=

WEB_SEARCH_PROVIDER=serper
SERPER_API_KEY=
GOOGLE_MAPS_API_KEY=
GOOGLE_MAPS_URL_SIGNING_SECRET=
BAIDU_MAPS_API_KEY=
LOG_LEVEL=INFO
```

YAML files under `configs/` contain only non-sensitive settings such as tool enablement, class paths, cost level, max calls, and whether confirmation is required.

Model roles:

- `brain`: multimodal ReAct controller. It reads the source image on the initial
  turn and also handles targeted visual reanalysis and map verification. Uses
  `BRAIN_*`.
- `memory_manager`: extracts reusable strategy memories from the full episode trajectory of completed verified episodes. Uses `MEMORY_MANAGER_*`, then falls back to `BRAIN_*`.

`configs/models.yaml` stores non-sensitive defaults such as role names, default
model names, temperature, and max token limits. There is no separate VLM role or
`VLM_*` configuration; all visual model calls use the Brain configuration.

## Run Demo

```bash
python scripts/run_single_image.py --image-path path/to/image.jpg --query "Where is this image taken?"
```

The output includes visual cues, hypotheses, selected tools, Web/POI/map-verification results, final answer, confidence, uncertainty radius, and tool trace.

For a live trace with colored Agent/Tool/Model/Workflow logs and streamed model output:

```bash
python scripts/run_single_image.py --image-path path/to/image.jpg --query "Where is this image taken?" --debug
```

You can also set `LOG_LEVEL=DEBUG` in `.env` to enable the same debug trace without passing `--debug`.
Use `--no-color` when you want copyable logs without ANSI color codes.

Runtime logs are also written to `logs/YYYY-MM-DD/log.txt`, for example `logs/2026-05-20/log.txt`. The directory and file name are configured in `configs/system.yaml` under `logging.file_root` and `logging.file_name`.

To clean an already copied debug log:

```bash
python scripts/clean_log.py demo_logs.txt
```

## Dataset Batch Evaluation

`scripts/run_dataset_eval.py` supports GeoExp7K, Im2GPS3K, and IMAGEO-Bench dataset2, runs a non-interactive workflow, and writes one CSV row per image. The CSV separates prediction levels into `pred_continent`, `pred_country`, `pred_region`, `pred_city`, `pred_street`, `pred_poi`, and `pred_coordinates`, with longitude and latitude in `pred_longitude` / `pred_latitude`.

Run a small smoke evaluation first:

```bash
python scripts/run_dataset_eval.py --datasets geoexp7k-test --limit 5 --output outputs/geoexp7k_smoke_eval.csv
```

Run all supported datasets with resumable output and full state traces:

```bash
python scripts/run_dataset_eval.py --datasets all --resume --trace-dir outputs/eval_traces --output outputs/dataset_eval_results.csv
```

Missing local image files are skipped by default; pass `--write-missing` if you want missing-image metadata rows in the CSV.

## Direct-VLM Batch Baselines

`scripts/run_batch_vlm_eval.py` evaluates a vision-language model directly from
the image and a fixed prompt. It does not construct the GeoAgent workflow and
does not expose tools, agents, or external memory. Dataset discovery, ground
truth fields, distance thresholds, and the main CSV columns are shared with
`run_dataset_eval.py`, while provider/model/job IDs and the raw model output are
added for auditability.

Alibaba Cloud Model Studio is the first implemented provider. Its Batch API uses
the OpenAI-compatible Files/Batches workflow and currently limits one input
JSONL file to 50,000 requests / 500 MB and one line to 1 MB. Configure `.env`:

```dotenv
BATCH_PROVIDER=aliyun
BATCH_API_KEY=your-model-studio-key
BATCH_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BATCH_MODEL=qwen3-vl-plus
```

Run the stages separately so a long asynchronous job can be resumed:

```bash
python scripts/run_batch_vlm_eval.py --action prepare --datasets imageobench-dataset2 --limit 100 --run-dir outputs/batch_vlm/qwen3_vl_plus
python scripts/run_batch_vlm_eval.py --action submit --run-dir outputs/batch_vlm/qwen3_vl_plus
python scripts/run_batch_vlm_eval.py --action status --run-dir outputs/batch_vlm/qwen3_vl_plus
python scripts/run_batch_vlm_eval.py --action wait --run-dir outputs/batch_vlm/qwen3_vl_plus
python scripts/run_batch_vlm_eval.py --action collect --run-dir outputs/batch_vlm/qwen3_vl_plus
```

`--action run` performs all five stages. `prepare` is offline and does not need
an API key. Every remote file ID, batch ID, status, output/error file, prompt
snapshot, and non-secret request setting is persisted in `manifest.json`.
Repeated `submit` calls skip chunks that already have remote IDs.

Local images use Base64 by default. Because Base64 expands the request and the
Alibaba input line limit is 1 MB, `BATCH_IMAGE_MAX_EDGE=2048` explicitly
re-encodes local inputs as JPEG and downsizes a longer edge above 2048 pixels;
the edge and quality settings are recorded in the manifest. For an exact,
unmodified input image, host the dataset on HTTP(S) or OSS and use:

```dotenv
BATCH_IMAGE_MODE=url
BATCH_IMAGE_BASE_URL=https://your-image-host.example/datasets
```

The URL is formed from the image path relative to `--dataset-root`. Oversized,
missing, or unreadable inputs are never silently dropped: they are written as
`input_error` rows and excluded from submission. Set `BATCH_ENABLE_THINKING=false`
when a supported Qwen model would otherwise enable paid thinking tokens.

For another vendor that implements the same OpenAI Files/Batches protocol, set
`BATCH_PROVIDER=openai_compatible` and change `BATCH_API_KEY`,
`BATCH_BASE_URL`, `BATCH_MODEL`, and the configurable endpoint/path/auth fields.
A vendor with a different batch protocol needs a new adapter in
`geoagent/batch/provider.py`; changing only the URL is not sufficient.

## External Experience Memory

The optional memory subsystem learns reusable geolocation strategies and failure
warnings, not place-specific facts. SQLite is authoritative and Chroma is the
required local persistent retrieval index whenever memory is enabled. Chroma
initialization, write, and query failures stop the run instead of silently
changing retrieval behavior. Memory is disabled by default so baseline runs
remain comparable.

Enable learning and retrieval for a single image:

```bash
python scripts/run_single_image.py --image-path path/to/image.jpg --memory-mode full --memory-dir outputs/my_experiment/memory
```

For experience-order dataset experiments, use `--memory-mode full --workers 1`.
Parallel workers do not provide a deterministic see-one-example-then-update order.
Use `retrieve_only` for a frozen-memory evaluation and `learn_only` to populate a
store without exposing retrieved memories to Brain. In `retrieve_only` and `full`,
the React workflow retrieves stored strategy memories directly through
`MemoryManager` at `post_initial_reasoning` and `pre_final_answer`; configure
either checkpoint independently under `memory_checkpoints` in
`configs/workflows.yaml`. A successful checkpoint retrieval gives Brain a revised
decision in the same workflow step. Set both flags to `false` to disable retrieval
for the task.

`MemoryManagerAgent` reviews each completed verified episode and may propose one
reusable strategy memory. Deterministic code validates the proposal, merges it
into a sufficiently similar memory of the same type, or writes it directly as a
new memory. There is no activation, dormancy, forgetting, archival, or periodic
maintenance. Brain reports the IDs of memories that affected a decision so
retrieval, return-to-Brain, and citation remain available for post-hoc analysis.

Because the simplified store has a new schema, start new experiments with an
empty memory directory rather than reusing a lifecycle-era SQLite database.

## Workflows

`react` is the only workflow. It runs VLM perception first, then enters an LLM-driven loop:

1. `BrainAgent` reads compact state and decides the next action.
2. `ReactWorkflow` validates and executes the requested tool.
3. The tool result is added to the conversation and Brain decides how it changes the hypotheses, confidence, and next action.

`BrainAgent` uses a multi-turn chat history rather than isolated single-shot prompts. The current structured state remains the authoritative fact snapshot, while recent Brain decisions and tool observations are passed back as conversation messages on the next turn. Older messages are compacted into `brain_conversation_summary`; tune limits in `configs/system.yaml` under `brain_conversation`.

Final answers are hierarchical, but a successful localization must include both `lat` and `lon`. If Brain proposes an answer without either coordinate, the workflow stops immediately and marks the task `failed`. `max_steps` is an optional hard cost budget and defaults to no limit; if the configured budget is exhausted before Brain submits an answer, the workflow stops without an extra Brain call and marks the task `failed`. Confidence and `uncertainty_radius_m` are Brain estimates recorded in the result, not code-level stopping thresholds.

All public `lat`/`lon` fields in tool results, hypotheses, final answers, and evaluation CSVs are normalized to WGS84. Provider-native coordinates from Baidu are preserved only as `raw_lat`, `raw_lon`, and `raw_coordinate_system` inside tool results.

For visual follow-up, Brain can call `visual_reanalysis`. It sends the image back to the VLM with a focused question such as "inspect the upper-left street sign" or "check road markings and vehicle details". Targeted VLM reanalysis is the preferred second-pass visual tool.

Brain is constrained to the configured tool registry. If it requests an unknown tool, the workflow rejects that request before execution and returns an observation listing the available tools, so the next Brain turn can recover without calling a nonexistent API. Geocoding and reverse geocoding go through the unified `geocode` / `reverse_geocode` tools, which route providers internally and hide the details from the Brain: mainland China uses Baidu Maps (`BAIDU_MAPS_API_KEY`), international uses LocationIQ (`LOCATIONIQ_KEY`) with Google Maps Geocoding (`GOOGLE_MAPS_API_KEY`) as fallback on failure or empty results. POI search routes internally too: mainland China -> Baidu, international -> Serper Places.

Map verification tools are available for already geocoded candidates. `streetview_verify` checks street-view metadata first, then compares multiple headings in one VLM pass; it is best for facades, signs, storefronts, lane layout, and road-level details. `map_tile_verify` downloads and stitches map tiles, with `map_type=satellite` for terrain/water/large spatial relationships and `map_type=roadmap` for road names, road networks, and POI labels. Both tools write intermediate imagery to `outputs/map_cache/map_imagery/<task_id>/`. Google Street View uses `GOOGLE_MAPS_API_KEY`; URL signing uses `GOOGLE_MAPS_URL_SIGNING_SECRET` when configured. Baidu street view and Baidu/Google map tiles are best-effort tile/web endpoints and can be disabled in `configs/tools.yaml`.

When `.env` contains the role-specific Brain settings, Brain reads the image
directly and performs structured reasoning. Web search uses Serper
(`SERPER_API_KEY`). POI search routes internally: mainland China POIs use Baidu
Maps, international POIs use the lower-cost Serper Places endpoint (Google Places
semantics), so there is no separate place-details tool. `GOOGLE_MAPS_API_KEY`
remains for the Google Maps geocoding fallback, Street View, and map imagery.
Successful web searches, POI searches, and static webpage reads are cached in
`outputs/web_cache/search_cache.sqlite3`. `webpage_read` parses bounded public
HTML locally without a paid API or JavaScript execution. Brain never chooses a
provider for POI search or geocoding — routing is internal. Only map verification
tools (`map_tile_verify`, `streetview_verify`) still take a `map_provider`: use
`baidu` for mainland China imagery verification and `google` elsewhere.

## Add A Tool

1. Create a subclass of `BaseTool`.
2. Register it with `@tool_registry.register("tool_name")`.
3. Return all outputs as `ToolResult`.
4. Add a non-sensitive entry in `configs/tools.yaml` with `class_path`, `enable`, `cost_level`, and `max_calls_per_task`.

## Add An Agent

1. Create a subclass of `BaseAgent`.
2. Register it with `@agent_registry.register("agent_name")`.
3. Implement `run(state: GeoLocalizationState) -> AgentOutput`.
4. Keep API keys and environment access out of agents.
