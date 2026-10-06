# Home energy assistant

GridPFN Home connects household readings, a saved personalized controller, TabPFN forecasts, a bounded language-model planner, and a calendar/chat interface. The LLM selects tools; the tools calculate results and grounded templates state the numbers. It never changes physical devices.

[Quick start](../energy_assistant/README.md)

## Model → household app

```bash
python train.py --config configs/gridpfn.toml
python hems_assistant.py --household results/personalized/models/home_HOME_ID --llm qwen35_2b
```

The training entrypoint belongs to the research pipeline. Its exported `home_model.json` bundle references the completed run, selected checkpoint, forecast context and source/data receipts. The launcher validates those references and binds the exported home. The launcher prepares the full recorded history automatically. Open it only after research models and settings are frozen; earlier training-date replays use the final saved model and are retrospective demonstrations.

A completed cohort run can also be used directly:

```bash
python hems_assistant.py --household PATH_TO_COMPLETED_RUN --home HOME_ID --days 14 --llm qwen35_2b
```

The default replay scope is all complete recorded history, with the checkpoint declared by a home bundle (`best_feasible` for a bare run). `--split test` or `--split validation` restricts replay to that period; `--days N` keeps only its latest N days. The app opens on the latest recorded day. The loader preserves the run's calendar period, cohort scaling, encoder context, appliance service and predictive-feature identity. Raw state width and auxiliary forecast width are checked separately. Forecast-cache files and their manifests participate in cache identity. Training artifacts and caches are not modified.

A legacy unversioned home descriptor is also supported:

```json
{"run_dir":"/absolute/path/to/completed/run","home_id":27,"checkpoint":"best_feasible"}
```

**Model roles:** the saved PPO controller uses its checkpoint and auxiliary forecast features. When present, the exact saved study X/y context, seed, estimator count and pinned TabPFN weights reconstruct the household forecasting model on CPU. Otherwise the assistant explicitly uses a separate explanatory fit on earlier household readings. SHAP describes the forecast, not PPO decisions. The economic scheduler uses forecasts to compare possible plans; it is a distinct controller. CPU predictions are not claimed bitwise identical to saved study results.

Preparation and immutable evidence live in ignored `results/`; changed model/data/source hashes select a fresh cache. First preparation takes longer than reopening. `--saved-forecasts` uses prepared predictions instead of fitting the daily explanatory forecast on first request. Local chat remains CPU-only. Automatic chat setup currently targets Linux x64; on other platforms use a hosted provider or an existing compatible server.

## Privacy and data flow

**Local by default. Federated by design.** TabPFN inference, numerical energy tools,
SHAP explanations and the default Qwen chat run locally. Model installation requires
downloads; local inference does not require a hosted chat provider.

| Boundary | What crosses it |
|---|---|
| Simulated household clients → coordinator | Controller weights; training-only scale statistics provide a shared coordinate system. Forecast contexts and value models remain home-specific. |
| Assistant → optional hosted chat | Tool descriptions and the current question; follow-up requests can include two earlier questions. No automatic household-table or numerical-tool-result upload. Anything typed into a question is part of that question. |
| Assistant → browser / downloaded report | The selected home's readings and derived results. Reports can contain household information; sharing them is a user action. |
| Public walkthrough | Generated household inputs and real application/model responses. |

Federation currently runs as a one-machine research simulation, with local access
to the cohort inputs. It is not a deployed network of isolated households. There
is no secure aggregation or differential-privacy mechanism, so federated averaging
alone is not a formal guarantee against information leakage from model updates.

## What each answer means

| Question | Evidence | Limit |
|---|---|---|
| Explain the bill | Recorded appliance/PV traces × declared tariffs | Reconstructed accounting, not a utility invoice; no battery or peer settlement in this view. |
| Which devices cost most? | Gross appliance costs − solar used at home − export credits + fixed fees | Reconciles exactly. Shared solar is not arbitrarily credited to individual devices; a gross cost is not avoidable savings. |
| What could have changed? | Actual recorded activity shown alongside selected and alternative simulated plans | Savings compare matched simulations; do not subtract a simulated bill from the unmodified recorded trace and call that a fair gain. |
| When should appliances run? | A dated state branches into a forecast-based future at 00:00, 06:00, 12:00 or 18:00 | Future exogenous observations are replaced by forecasts; declared daily appliance needs remain known. This is not a live home connection. |
| Why this forecast? | Exact four-group marginal Shapley enumeration over eight training-background rows | Explains the TabPFN demand estimate, not physical causes or the PPO controller. Correlated-feature mixing can produce uncommon inputs. |

Planning uses TabPFN for fixed demand, PV and outdoor temperature. **Prices retain the observed base rate plus the known time-of-use schedule**; no TabPFN price-prediction claim is made. Battery charge and indoor temperature are projected by the simulator, not independently predicted by the foundation model. The economic coordinator is a constrained heuristic/LP controller, not a globally optimal oracle. Suggested savings require completed charging and laundry, no worse comfort, no lower terminal battery and no solver fallback against the declared reference. Forecast-based plans can still fail absolute comfort or service requirements; their actual checks remain visible.

A period comparison uses the same plan over the same available dates, with checks on every day. It never selects a different winner in hindsight per day or multiplies one day's savings into a monthly claim. Days have independent initial states; period totals are not continuous month-long simulations. Missing days are reported, never estimated. The colour scale is fixed across plans/dates and combines temperature deviation and duration, saturating at 24 degree-hours. Positive tile balances mean net energy credit; they do not imply investment profit.

Appliance categories follow the dataset's preprocessing, including grouped cooling and washing-related circuits. They are not a diagnosis of an individual appliance. Retrofit savings, hardware faults and live actuation remain unsupported.

## Chat providers

`--llm qwen35_2b` manages the pinned [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) model through [llama.cpp](https://github.com/ggml-org/llama.cpp). The [quantized artifact](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) is revision- and SHA-256-pinned; runtime files, weights and logs remain local. The bounded planner uses a 4,096-token context, one CPU slot and no thinking output. Downloads are reused and verified. A busy adjacent chat port produces an error instead of attaching to another app's server.

OpenAI uses [Responses function calling](https://developers.openai.com/api/docs/guides/function-calling); Claude uses [Messages tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools). Choose a model ID accessible to your account. Hosted availability is not implied by an accepted setting. Keys passed via environment remain in memory unless explicitly saved through Settings; Settings stores restricted local files (mode 0600), not encrypted secrets. Keys are never returned to the browser. Changing provider does not transfer its key. A failed/invalid tool call falls back to visibly labelled guided routing.

Only questions and recent question history go to the LLM; readings, checkpoint weights and calculated bills stay local. Questions themselves can contain personal information. Tools reject model-supplied household/date overrides, and the API is bound to one home. The app binds to loopback; forward port 8770 over SSH for a remote browser. A GitHub Pages site can show the diagram/video, but cannot run this Python backend.

## MCP

The same nine tools are available to an external MCP host; that host supplies its own LLM:

```bash
python -m energy_assistant mcp --evidence PATH_TO_PREPARED_EVIDENCE --home HOME_ID --live-tabpfn
```

```json
{
  "command":"/absolute/path/to/repo/.venv-assistant/bin/python",
  "args":["-m","energy_assistant","mcp","--evidence","/absolute/path/to/evidence","--home","27","--live-tabpfn"],
  "cwd":"/absolute/path/to/repo",
  "env":{"CUDA_VISIBLE_DEVICES":""}
}
```

`cwd` support depends on the host. Tools: `list_homes` (the bound home only), `inspect_home`, `explain_bill`, `appliance_breakdown`, `forecast_and_explain`, `compare_schedules`, `schedule_comparison`, `plan_day`, `export_plan`. Date scopes support `end_date`; plan tools also accept the reference plan, and `plan_day` accepts a planning hour. Numerical facts and plain-language notes are the default; full reports retain provenance and hashes.

## TabPFN features

- Real supervised household context, separated chronologically from queried days.
- Local `fit_with_cache` reused for inference and explanation queries.
- Exact grouped Shapley values implemented in a small audited routine, not the Prior Labs `shapiq` adapter. This explanation method is also applicable to other models.
- Hosted Thinking mode is not enabled or claimed. It is separate from an LLM's reasoning switch and requires its own controlled forecasting experiment.

[Interpretability](https://docs.priorlabs.ai/capabilities/interpretability) · [KV cache](https://docs.priorlabs.ai/capabilities/kv-cache) · [Thinking mode](https://docs.priorlabs.ai/capabilities/thinking-mode)

```bash
python -m pytest tests/test_energy_assistant.py -q
```


### Forecast explanations and energy exchange

In a day plan, expand **Why this forecast? · TabPFN SHAP**, choose everyday demand,
solar output or outdoor temperature, then select a future hour. The app shows the
last four available readings, a forecast point and a waterfall from the training
background to the prediction. The readings table exposes the physical inputs
behind the latest, previous and recent-average features. At midnight, only the
midnight reading is used. Explanations beyond the trained horizon are rejected.

Attributions enumerate all 16 coalitions of four feature groups over eight fixed
training examples. Their sum is checked against the prediction. These are
model-agnostic marginal Shapley values applied to TabPFN, not causal effects or an
explanation of PPO actions. Group substitution can create unlikely combinations;
the chart does not show calibrated uncertainty. Temperature uses degrees Celsius,
not kWh. Demand excludes separately controlled appliances. Outdoor temperature
comes from observed history, not an external weather forecast.

Saved study context and weights are reused where present. CPU recalculation can
differ from saved predictions; the interface displays both if their difference
exceeds 0.001 in the target's units. A fallback explanatory model is explicitly
identified when study context identity is unavailable. Results are cached for up
to 128 distinct date/origin/target/variable requests. No extra calls to the chat
provider are needed for these explanations.

MCP exposes `explain_forecast_inputs(home, date, origin, target, variable)` using
the same bounded calculation as the app. The HTTP endpoint is
`/api/explanation/{home}/{date}?origin=6&target=9&variable=solar`.

**Home ↔ Grid** shows imports, exports and credits derived from the recorded
balance, and projected flows when a prepared outlook contains its simulator
ledger. Old bundles without that ledger explicitly report the gap. Peer trading
is disabled in this single-home assistant: there is no neighbour demand, matching,
settlement or trading agent. Grid export credits are not evidence of peer trades
or an optimal selling policy. Battery discharge serves on-site demand and cannot
create exports in the current environment. No live transactions are executed.
