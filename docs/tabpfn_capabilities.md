# TabPFN's role

| Capability | Implemented role |
|---|---|
| TabPFN-3.5 supervised regression | Local six-hour forecasts of load, solar and temperature |
| In-context adaptation | Actual preceding labeled examples from the particular home |
| Frozen foundation weights | Pinned checkpoint and hash; one estimator per target |
| Controller integration | Forecasts feed the same PPO architecture used by all comparison arms |
| Local execution | Household observations stay in local files during inference |
| Interpretation | Inspect forecast sensitivity, schedule constraints, bills and comfort separately |

The predictor uses the estimator's supervised fit/predict interface. The current
submission does not use artificial-label embeddings. Only controller parameters
are trained with PPO/FedAvg; the TabPFN backbone is not fine-tuned. Thinking mode
and TabTune are not used, so no gains are attributed to them.

A model explanation is useful only when it refers to the exact predictor, context
and query used. The assistant's live forecaster must reuse the exported context to
explain the studied predictor. Forecast attribution alone does not explain PPO,
prove causality or make TabPFN inherently more interpretable than every baseline.
Matching context and weights is not a numerical parity guarantee: live SHAP
explains the recalculated forecast shown in the app, with any stored-study value
identified separately. CPU and GPU inference can differ numerically.
The bill calculation and physical action replay provide a separate, checkable
explanation of the simulated schedule. See [architecture](architecture.md) and
[home model contract](MODEL_BUNDLE.md).

Forecast accuracy and control quality are distinct endpoints. All six comparison
arms remain in the evidence, whether TabPFN wins or loses. Foundation pretraining,
model sizes and compute differ; matching context rows and policy architecture
does not make this an equal-compute study.
