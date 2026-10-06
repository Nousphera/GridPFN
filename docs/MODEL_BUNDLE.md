# Local home model contract

`python train.py --config configs/gridpfn.toml` runs the existing seasonal pipeline
for **TabPFN + FedAvg across all 25 homes**, selecting its training budget on the
last seven days of September, then refitting from scratch on May–September.
The model settings and convergence rule come from `configs/seasonal.json`.
This command does not score October or run comparison baselines. It requires the
original locally licensed dataset and TabPFN environment; this is substantial
training, not an instant demo. `--export-only` exports an already completed run.
Before training, the launcher checks the pinned TabPFN-3.5 regression checkpoint
in the official model cache. If missing, it uses TabPFN's official download and
authentication flow, then verifies its SHA-256. Model access and license acceptance
remain required; a mismatched cached file fails rather than being silently replaced.
`--export-only` does not download model weights.

The TOML tariff section declares a fixed original simulator preset. The launcher
passes its five training-fitted time-of-use blocks, enabled peer trading, demand
response limit, export cap and fees explicitly into the frozen training arguments;
it rejects economic overrides in the base protocol. Changing these settings to
improve a method's reported ranking is not part of this preset.

Each `results/personalized/models/home_ID/` contains `home_model.json` and `model.sha256`.
This is a **local-reference bundle, not a portable model**. Keep the original
checkout, run artifacts, forecasts, weather and cohort CSV files in place.
Reproducing shared scaling requires the entire training cohort, even when serving
one home. FedAvg shares the actor; forecasts, context and local critics are
home-specific. This is not a formal privacy guarantee or a federated forecaster.

Application integration:

```python
from gridpfn.deployment import load_model_bundle

model = load_model_bundle(model_directory)  # verifies every local dependency
home = model["home_id"]
run = model["source_run"]
checkpoint = model["checkpoint_name"]  # latest, the fixed refit checkpoint
data_period = model["data_period"]     # month_10_refit
context_path = model["files"]["forecast_context"]["path"]
```

The assistant accepts this contract through `--model MODEL_DIR` and binds the
service to exactly `home`. It preserves `data_period`, checks the actor's 17 raw
inputs and separate auxiliary width 32, and verifies the per-home feature
fingerprint. Descriptor and context integration have been checked on all 25
completed October refit bundles. All held-out policy replays passed, and the
official study export matches the descriptors used for final assistant checks.

Launch an exported research model:

```bash
python hems_assistant.py --household results/seasonal_20261006/models/home_27 --llm guided
```

Open `http://127.0.0.1:8770`. Use `--llm qwen35_2b` for the supported local language
model, or follow the [assistant setup](../energy_assistant/README.md) for a hosted
provider. The home is chosen by the model directory; it is not selected in the app.

The context NPZ is the final training block's exact `X`/`y` supervised examples,
with recorded seed, row count, estimator count, horizon and feature schema.
Use only `X`/`y` to fit the corresponding TabPFN adapter; its `query` arrays belong
to the recorded future replay and are not online context. A separately fitted
assistant predictor must be labelled separately from the study model.

Reusing the same context and checkpoint does not establish numerical parity with
stored forecasts. Live attributions must reconstruct the displayed recalculated
prediction; preserve and distinguish the stored study prediction when it differs.
CPU and GPU inference can differ numerically. The benchmark continues to use its
immutable recorded forecasts; live SHAP explains the displayed recalculation.

Preparing October replay opens held-out observations: defer until all research
models are frozen. A single-home replay without peer settlement is not the cohort
benchmark. No physical device actuation is included.

Absolute dependency paths and SHA-256 receipts record inputs, source, prior
selection and final checkpoint. Verification detects drift; it does not
authenticate files from an untrusted publisher. Load only trusted locally
produced checkpoints. Credentials are not included.

## Where each home's model and history live

- Production training command: `results/personalized/models/home_ID/`.
- Completed research study: `STUDY_DIR/models/home_ID/`, exported by
  `python -m gridpfn.experiments.finalize_study STUDY_DIR` after all monthly refits,
  oracle certificates, policy evaluations and the verified report are complete.
- Original local measurements: `dataset/split_homes_clean/home_ID.csv`; shared
  weather and prices: `dataset/temp_price_newyork.csv`.

The original data cover May–October 2019. Browsing that entire recorded history
does not make it all held-out evaluation data. The October model trains through
September; only October is held out for that model. Earlier monthly models have
their own chronological training and evaluation periods.

The assistant prepares forecasts and policy replays for all complete recorded
days by default, with separate training and held-out labels. `--split test` or
`--split validation` restricts replay; bills remain available across recorded
history. Replaying May with the final September-trained model
is a retrospective demonstration that uses later training knowledge, not a
causal May backtest. Use the corresponding earlier frozen monthly model for
reported forward evaluation. Fitting an additional deployment model on all
May–October data is a separate post-benchmark extension; it does not replace the
frozen models or their held-out results.

For calendar integration, `model["history"]` provides ordered `available_dates`,
`training_dates`, and `evaluation_dates` (the held-out month), with exclusive
training/evaluation cutoffs. `model["files"]["home_data"]` gives the bound home's
raw CSV path and SHA-256; `home_history` identifies the saved chronological archive.
The exporter reads only that archive's date coordinates, not its physical values
or outcomes, and verifies disjoint chronological partitions and the forecast
context's training dates. Calendar entries describe available recorded history;
only `evaluation_dates` qualify as held out for this particular final policy.
