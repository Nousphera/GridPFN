# Seasonal comparison protocol

**Question:** do better forecasts improve the same home-energy controller?
The specification is [configs/seasonal.json](../configs/seasonal.json).
Every completed method and month is retained, including unfavorable results.

| Shared condition | Specification |
|---|---|
| Cohort | All 25 available homes; one seed, 41 |
| Forward test folds | June through October 2019 |
| Methods | TabPFN-3.5, TabFM, TabICLv2, Extra Trees, persistence, history only |
| Supervised context | Up to 1,024 identical sampled prior labeled rows |
| Forecast horizon | Six hourly steps: load, solar and outdoor temperature |
| Controller | 256-unit actor / 64-unit private value head; FedAvg |
| Initialization | 60 feedback-imitation rounds; shuffled real days; no synthetic augmentation |
| PPO selection | Learning rate 0.00003; validation every 100 episodes; cap 20,000 |

## Forward selection and final refit

For each target month, the preceding seven calendar days select the training
budget. Selection training begins May 1 and excludes that validation week. The
maximum validation reward determines the selected episode, including zero when
initialization is best. A fresh model is then fitted for that budget using **all
preceding training and validation dates**. Refits perform no checkpoint selection
or evaluation; only their final checkpoint is eligible for the monthly test.

All 30 final refits are frozen together before new test metrics are inspected.
An earlier observed month can train a later fold, as in monthly retraining.
October never trains a reported model. June–August have historical development
exposure; temporal separation does not erase that fact. Available data span
May–October, so this is not a winter or year-round evaluation.

Forecast contexts use only complete days preceding a query block, refreshed every
28 days. The first 14 training days use persistence. Final predictors sample from
the full eligible prior pool, subject to the 1,024-row cap; they do not consume
every historical row. Scalers and tariff normalization fit the corresponding
training pool, expanded to training plus validation during final refit.

Selection stops after at least 3,000 episodes when the declared 20-check
validation plateau and reward/comfort/cost/import stability criteria hold, or at
the cap. A capped run is not called converged. A plateau does not prove a global
optimum. Methods share the stopping rule and architecture but may select different
budgets; this is not an equal-compute comparison.

## Outcomes and uncertainty

The report shows combined simulator objective, bill before demand-response credits,
comfortable time, violation hours and squared discomfort. It averages days within
each home, then reports mean and sample SD across the 25 home means. SD describes
household heterogeneity, **not a confidence interval, seed uncertainty or unseen-home
generalization**. Monthly and pooled results are retained. Forecast errors are
reported separately from control outcomes.

Test dates must be complete for the full cohort. Home 27 lacks July 29, which is
excluded from common control evaluation. All available complete dates remain
eligible for their own home's training; missing dates are not invented.

## Oracle and verification

The oracle sees perfect future traces and minimizes the same community objective
under the original constraints. Numerical lower/upper bounds and independent
replay are required. The bound applies to the **combined objective**; oracle bills
and comfort are components of that schedule, not separate optimum bounds. A
deployable controller cannot access this perfect-future information.

Selected recipes, heads, contexts, data, model versions and numerical source are
hashed before scoring. Each policy is independently replayed against pinned
original physics. Reporting rejects incomplete runs, changed artifacts, missing
methods and inconsistent dates. Better forecast accuracy alone does not establish
better control; this cohort and seed cannot establish SOTA or real-home savings.
