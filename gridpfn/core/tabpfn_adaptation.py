"""Small numeric regression prompts, adapted once and frozen before control."""

import time

import numpy as np
import torch


class PromptRegressor:
    def __init__(self, device, seed=6, context=64, steps=12):
        if context < 1 or steps < 0:
            raise ValueError("Prompt context must be positive and adaptation steps nonnegative")
        self.device, self.seed, self.context, self.steps = device, seed, context, steps

    def _mean(self, x):
        logits, _, borders = self.model.forward(x, use_inference_mode=True)
        logits = logits.T.float()
        borders = torch.as_tensor(borders[0], device=logits.device, dtype=logits.dtype)
        centers = (borders[:-1] + borders[1:]) / 2
        return (logits.softmax(-1) * centers).sum(-1) * float(self.model.y_train_std_) + float(
            self.model.y_train_mean_
        )

    def fit(self, x, y, groups):
        # Entire dates, never random interleaved hours, separate context,
        # adaptation queries and internal validation within the earlier prefix.
        days = np.unique(groups)
        first, last = max(1, len(days) // 2), max(2, 3 * len(days) // 4)
        context_rows = np.flatnonzero(groups < days[first])
        queries = np.flatnonzero(
            (groups >= days[first]) & (groups < days[min(last, len(days) - 1)])
        )
        validation = np.flatnonzero(groups >= days[min(last, len(days) - 1)])
        if not len(queries) or not len(validation):
            raise ValueError("Prompt adaptation requires at least four earlier complete dates")
        rng = np.random.default_rng(self.seed)
        selected = rng.choice(context_rows, min(self.context, len(context_rows)), replace=False)
        denominator = max(float(np.std(y[context_rows])), 1e-3)
        if np.std(y[context_rows]) <= 1e-12:
            self.constant = float(np.mean(y[context_rows]))
            error = float(np.mean(((y[validation] - self.constant) / denominator) ** 2))
            self.metrics = {
                "constant_context": True,
                "initial_validation_mse": error,
                "best_validation_mse": error,
                "steps": 0,
                "seconds": 0,
                "parameters": 0,
                "history": [error],
            }
            return self
        if np.std(y[selected]) <= 1e-12:
            selected[:2] = context_rows[[np.argmin(y[context_rows]), np.argmax(y[context_rows])]]
        from tabpfn import TabPFNRegressor

        self.model = TabPFNRegressor(
            device=self.device,
            n_estimators=1,
            random_state=self.seed,
            differentiable_input=True,
            inference_precision=torch.float32,
            ignore_pretraining_limits=True,
        )
        self.x = torch.tensor(
            x[selected], device=self.device, dtype=torch.float32, requires_grad=True
        )
        self.y = torch.tensor(
            y[selected], device=self.device, dtype=torch.float32, requires_grad=True
        )
        self.model.fit_with_differentiable_input(self.x, self.y)
        for model in self.model.models_:
            model.requires_grad_(False)
        optimizer = torch.optim.Adam([self.x, self.y], lr=0.01)
        check = rng.choice(validation, min(128, len(validation)), replace=False)
        xv = torch.tensor(x[check], device=self.device, dtype=torch.float32)
        yv = torch.tensor(y[check], device=self.device, dtype=torch.float32)
        started = time.perf_counter()
        with torch.no_grad():
            initial = float(((self._mean(xv) - yv) / denominator).square().mean())
        best, best_x, best_y = initial, self.x.detach().clone(), self.y.detach().clone()
        records = [initial]
        for step in range(self.steps):
            selected = rng.choice(queries, min(32, len(queries)), replace=False)
            batch = torch.tensor(x[selected], device=self.device, dtype=torch.float32)
            targets = torch.tensor(y[selected], device=self.device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            self.model.fit_with_differentiable_input(self.x, self.y)
            loss = ((self._mean(batch) - targets) / denominator).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.x, self.y], 1)
            optimizer.step()
            self.model.fit_with_differentiable_input(self.x, self.y)
            with torch.no_grad():
                error = float(((self._mean(xv) - yv) / denominator).square().mean())
            records.append(error)
            if error < best:
                best, best_x, best_y = error, self.x.detach().clone(), self.y.detach().clone()
        self.x, self.y = best_x, best_y
        self.model.fit_with_differentiable_input(self.x, self.y)
        self.metrics = {
            "initial_validation_mse": initial,
            "best_validation_mse": best,
            "history": records,
            "steps": self.steps,
            "seconds": time.perf_counter() - started,
            "parameters": self.x.numel() + self.y.numel(),
        }
        return self

    @torch.no_grad()
    def predict(self, x):
        if hasattr(self, "constant"):
            return np.full(len(x), self.constant)
        return np.concatenate(
            [
                self._mean(torch.tensor(chunk, device=self.device, dtype=torch.float32))
                .cpu()
                .numpy()
                for chunk in np.array_split(x, max(1, (len(x) + 127) // 128))
            ]
        )
