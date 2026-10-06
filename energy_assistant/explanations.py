"""Small, exact grouped Shapley audit of a predictor, with a fixed background.

This is model-agnostic marginal SHAP, not a causal explanation or a claim that
TabPFN is inherently interpretable. Sixteen coalitions make the audit bounded.
"""

import math

import numpy as np

GROUPS = {
    "Recent household demand": [0, 4, 8],
    "Recent solar generation": [1, 5, 9],
    "Weather and current tariff": [2, 3, 6, 7, 10, 11],
    "Time and forecast horizon": [12, 13, 14, 15],
}


def grouped_shapley(predict, query, background):
    """Exactly enumerate groups; integrate over the supplied empirical background."""
    query = np.asarray(query, dtype=np.float32)
    background = np.asarray(background, dtype=np.float32)
    if query.shape != (16,) or background.ndim != 2 or background.shape[1] != 16:
        raise ValueError("Expected a 16-feature query and a matching background")
    if not len(background) or not np.isfinite(background).all() or not np.isfinite(query).all():
        raise ValueError("Explanation inputs must be finite and nonempty")
    groups = list(GROUPS.items())
    count = len(groups)
    rows = []
    for mask in range(1 << count):
        hybrid = background.copy()
        for index, (_, columns) in enumerate(groups):
            if mask & (1 << index):
                hybrid[:, columns] = query[columns]
        rows.append(hybrid)
    values = np.asarray(predict(np.concatenate(rows)), dtype=float)
    if values.shape != ((1 << count) * len(background),) or not np.isfinite(values).all():
        raise ValueError("Predictor returned invalid explanation values")
    coalition = values.reshape(1 << count, len(background)).mean(1)
    contributions = []
    for index, (name, _) in enumerate(groups):
        attribution = 0.0
        for mask in range(1 << count):
            if mask & (1 << index):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            attribution += weight * (coalition[mask | (1 << index)] - coalition[mask])
        contributions.append({"feature": name, "kwh": float(attribution)})
    return {
        "method": "Exact four-group marginal Shapley values over a fixed training background",
        "baseline_kwh": float(coalition[0]),
        "prediction_kwh": float(coalition[-1]),
        "contributions": contributions,
        "additivity_error": float(
            abs(coalition[0] + sum(c["kwh"] for c in contributions) - coalition[-1])
        ),
        "background_rows": len(background),
        "model_queries": len(values),
        "caveat": "Feature influence on this forecast, not causes of your bill. Mixing feature groups may create uncommon inputs. This does not explain the PPO controller.",
    }
