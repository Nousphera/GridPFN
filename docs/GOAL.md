# Release goal

Deliver **GridPFN**: a local, home-specific energy assistant backed by TabPFN-3.5
forecasts and a FedAvg-trained controller, with reproducible evidence and a clear
train → export → serve workflow.

Completion requires:

1. Six methods across all 25 homes and five forward months; prior-week selection,
   then final refits on all preceding training and validation dates.
2. All final checkpoints frozen before scoring; certified perfect-future oracles
   and independent original-physics replay for every evaluated policy.
3. Complete results, including losses, with bill/comfort/objective figures and
   SD across homes, plus clear oracle and temporal-split explanations.
4. A verified per-home model export and assistant interface; exact forecast
   context, model identity and date provenance remain visible.
5. Compact instructions, process diagram and results explorer; obsolete public
   presentation removed while private evidence and active app work are preserved.
6. Owned code tested, reviewed, committed and published at Nousphera/GridPFN with the live project website.

Code is Apache-2.0. Household inputs and model weights retain separate terms.
Public release is authorized. Do not submit the entry form automatically. A source-only archive must exclude raw inputs,
weights, credentials, research notes and old Git history.
