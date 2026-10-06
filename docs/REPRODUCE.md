# Run and reproduce

Use Python 3.12. Household CSV files, credentials, weights and runs remain local.
Obtain authorized [input data](../dataset/README.md) and
[TabPFN model access](https://docs.priorlabs.ai/models/accessing-model-weights).

```bash
git clone https://github.com/Nousphera/GridPFN.git
cd GridPFN
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-assistant.txt
```

## Train a home model, then launch its assistant

```bash
python train.py --config configs/gridpfn.toml
python hems_assistant.py --household results/personalized/models/home_27 --port 8770
```

Training runs TabPFN + FedAvg across all 25 homes, chooses its training budget on
late-September validation, and refits on May–September. It exports one verified
local-reference bundle per home. Training is substantial; this is not a quick
smoke run. Completed runs can be exported using `train.py --config
configs/gridpfn.toml --export-only`. Keep the original run and input files in place.
[Bundle contract and locations](MODEL_BUNDLE.md).

The assistant loads the verified home bundle and its saved forecast context.
Its default local LLM runs on CPU; `--llm guided` requires no LLM. See [assistant setup](../energy_assistant/README.md) for hosted providers,
MCP and the generated-data demonstration.

## Reproduce the complete comparison

Optional research predictors run in separate environments because their package
requirements differ. TabFM's weights have noncommercial/nonproduction terms.

```bash
python3.12 -m venv .venv-tabfm
.venv-tabfm/bin/python -m pip install -r requirements-tabfm.txt
python3.12 -m venv .venv-tabicl
.venv-tabicl/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv-tabicl/bin/python -m pip install -r requirements-tabicl.txt
```

Initialize the authorized TabPFN cache once before the research runner:

```bash
python -c "from gridpfn.core.model import _tabpfn_backbone; _tabpfn_backbone('cpu')"
```

Run all selection and train-plus-validation refits, then independently evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 python -m gridpfn.experiments.seasonal_study \
  --output results/seasonal \
  --protocol configs/seasonal.json \
  --tabfm-python .venv-tabfm/bin/python \
  --tabicl-python .venv-tabicl/bin/python

CUDA_VISIBLE_DEVICES='' python -m gridpfn.experiments.finalize_study \
  results/seasonal --workers 4
```

Use an available GPU; forecasting is the GPU workload and policy training runs on
CPU. The declared study covers all six methods across five monthly forward folds,
with one seed. The [protocol](protocol.md) specifies the temporal split, stopping
rule, original physics and reporting limits. Do not change settings in an existing
output directory; new protocols need new outputs.

On a large workstation, overlap the monthly folds by adding
`--fold-workers 5 --policy-workers 16 --gpu-workers 3 --cpu-forecast-workers 6`
to the study command. These are concurrency limits; reduce them to fit available
memory. Each fold still finishes selection before refitting, and training settings
remain identical. On Linux, the scheduler can adopt existing workers from the same
output directory without restarting them; `scheduler.jsonl` records admission and
completion. Only one scheduler may own that directory.

The finalizer refuses incomplete training. It freezes all 30 final checkpoints
before reading test outcomes, computes certified monthly perfect-future oracles,
evaluates and independently replays every policy, and exports the comparison.
It then exports October TabPFN home bundles under `results/seasonal/models/`.
Interrupted oracle/evaluation work is reusable only when its receipts still match.
Incomplete training is preserved for inspection, not silently overwritten.

## View and verify

```bash
python demo.py
# Open http://127.0.0.1:8767/performance.html

python -m pip install pytest ruff
python -m pytest -q
ruff check .
node --check site/performance.js
```

The static explorer reads the same generated `site/performance.json` as the SVG,
PNG and PDF figures. It requires no household data or model credentials. The
source tests cover chronological contexts/refits, model adapters, artifact
identity, original physics and report arithmetic. Optional model/GPU checks have
separate prerequisites; tests do not establish the final experiment's outcome.

## Private release and hosting

Code is Apache-2.0; weights and household inputs retain separate terms. Real-data
scores need authorized original inputs. The generated assistant demonstration
uses fictional data and does not reproduce those recorded values.

The repository remains private as requested. Older Git history contains household
CSVs, so public release needs a rights decision or a sanitized new repository.
After a reviewed commit, `python -m scripts.make_release` creates a source-only
archive under `results/release/` with file hashes and no Git history, raw data,
weights, credentials or research notes.

## Project website

The [project website](https://nousphera.github.io/GridPFN/) and interactive results
are published from this repository by the Pages workflow on every push to `main`.
Only the reviewed static payload is deployed:

```bash
python -m scripts.build_site --output results/site-public
```

The website includes aggregate evidence, diagrams and the generated-home
walkthrough. Household input traces, weights and private research stay local.
TabPFN inference and the Python assistant run on your own machine.
