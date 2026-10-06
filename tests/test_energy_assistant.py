"""Financial, explanation, agent-boundary and HTTP contracts for the demo."""

import hashlib
import io
import json
from copy import deepcopy

import numpy as np
import pytest
from fastapi.testclient import TestClient

from energy_assistant.agent import chat_events, llm_tools
from energy_assistant.explanations import grouped_shapley
from energy_assistant.service import EvidenceService
from energy_assistant.simulation import ledger_row
from energy_assistant.web import create_app


@pytest.fixture
def bundle(tmp_path):
    hourly = [ledger_row(h, 2, 0, 0.2, 0.025, 0.01) for h in range(24)]
    baseline = {
        "bill": 9.84,
        "comfort_pct": 100.0,
        "degree_hours": 0.0,
        "ev_complete": True,
        "washer_complete": True,
        "terminal_battery_soc": 0.5,
        "solver_failures": 0,
        "limits": "Historical simulation",
        "actions": [],
        "ledger": hourly,
    }
    better = baseline | {"bill": 8.0}
    forecast = {"origin_hour": 12, "target_hour": 18, "rows": [], "explanation": None}
    data = {
        "schema_version": 1,
        "created_at": "2026-10-06",
        "provenance": {"kind": "synthetic"},
        "execution": "prepared",
        "currency": "USD",
        "forecaster": "Seasonal",
        "model": {"backend": "seasonal"},
        "homes": {
            "101": {
                "training_days": 47,
                "train_start": "2019-06-01",
                "train_end": "2019-07-17",
                "bill": {
                    "days": [{"date": "2019-07-31", "total": 9.84, "hours": hourly}],
                    "period_total": 9.84,
                    "import_cost": 9.6,
                    "export_credit": 0.0,
                    "fixed_cost": 0.24,
                    "note": "Generated",
                },
                "dates": {
                    "2019-07-31": {
                        "forecast": forecast,
                        "scenarios": {"feedback": baseline, "seasonal": better},
                    }
                },
            }
        },
    }
    write_bundle(tmp_path, data)
    return tmp_path


def write_bundle(path, data):
    raw = json.dumps(data).encode()
    (path / "evidence.json").write_bytes(raw)
    (path / "evidence.sha256").write_text(hashlib.sha256(raw).hexdigest())


def test_bill_units_and_export_credit():
    row = ledger_row(3, 2.5, 0.75, 0.24, 0.08, 0.15)
    assert row["net_cost"] == pytest.approx(0.69)
    with pytest.raises(ValueError):
        ledger_row(1, float("nan"), 0, 0.2, 0.02)


def test_source_receipt_tracks_implementations_beyond_unchanged_wrappers(tmp_path):
    from energy_assistant.prepare import source_hashes

    names = (
        "dataset.py",
        "environment.py",
        "forecasting.py",
        "economic_control.py",
        "em_strategy.py",
        "model.py",
        "gridpfn/core/environment.py",
        "gridpfn/foundation_backends/tabpfn_backend.py",
        "energy_assistant/prepare.py",
        "energy_assistant/trained_forecast.py",
        "energy_assistant/live.py",
    )
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original source\n")
    (tmp_path / "energy_assistant/.env").write_text("PRIVATE_PLACEHOLDER=not-source\n")
    before = source_hashes(tmp_path)
    assert set(before) == set(names)
    for name in ("gridpfn/core/environment.py", "energy_assistant/live.py"):
        (tmp_path / name).write_text("# changed implementation\n")
    after = source_hashes(tmp_path)
    assert {name for name in before if before[name] != after[name]} == {
        "gridpfn/core/environment.py",
        "energy_assistant/live.py",
    }
    (tmp_path / "gridpfn/foundation_backends/tabpfn_backend.py").unlink()
    with pytest.raises(ValueError, match="Missing source implementations"):
        source_hashes(tmp_path)


def test_prepared_evidence_cache_tracks_backend_inputs_and_serving_options(tmp_path):
    from hems_assistant import evidence_cache_key

    checkpoint = tmp_path / "heads.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    options = dict(
        sources={"gridpfn/foundation_backends/tabpfn_backend.py": "initial-hash"},
        home=27,
        split="test",
        days=31,
        context=96,
    )
    initial = evidence_cache_key([checkpoint], **options)
    assert initial == evidence_cache_key([checkpoint], **options)
    changed_sources = {"gridpfn/foundation_backends/tabpfn_backend.py": "changed-hash"}
    assert initial != evidence_cache_key([checkpoint], **{**options, "sources": changed_sources})
    for key, value in (("home", 950), ("split", "validation"), ("days", 1), ("context", 128)):
        assert initial != evidence_cache_key([checkpoint], **{**options, key: value})
    checkpoint.write_bytes(b"different synthetic checkpoint")
    assert initial != evidence_cache_key([checkpoint], **options)


def test_grouped_shapley_recovers_linear_contributions():
    query = np.ones(16)
    weights = np.arange(1, 17)
    result = grouped_shapley(lambda x: x @ weights + 3, query, np.zeros((2, 16)))
    assert result["baseline_kwh"] == 3
    assert result["prediction_kwh"] == 139
    assert [r["kwh"] for r in result["contributions"]] == pytest.approx([15, 18, 45, 58])
    assert result["additivity_error"] < 1e-10


@pytest.mark.parametrize(
    "change",
    [
        {"comfort_pct": 99},
        {"degree_hours": 1},
        {"ev_complete": False},
        {"washer_complete": False},
        {"terminal_battery_soc": 0.49},
        {"solver_failures": 1},
    ],
)
def test_cheaper_plan_rejected_when_guard_fails(bundle, change):
    service = EvidenceService(bundle)
    assert service.compare_schedules("101", "2019-07-31")["recommended"] == "seasonal"
    service.data["homes"]["101"]["dates"]["2019-07-31"]["scenarios"]["seasonal"].update(change)
    result = service.compare_schedules("101", "2019-07-31")
    assert result["recommended"] is None
    assert result["candidates"][1]["reasons"]


def test_tampered_bundle_refused(bundle):
    (bundle / "evidence.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        EvidenceService(bundle)


def test_live_requires_context(bundle):
    with pytest.raises(ValueError, match="current TabPFN"):
        EvidenceService(bundle, live=True)


def test_http_stream_and_report(bundle, monkeypatch):
    monkeypatch.delenv("ENERGY_LLM_URL", raising=False)
    client = TestClient(create_app(bundle))
    response = client.post(
        "/api/chat", json={"home": "101", "date": "2019-07-31", "message": "Why is my bill high?"}
    )
    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert [e["event"] for e in events] == ["status", "tool_start", "tool_done", "answer"]
    assert "$9.84" in events[-1]["text"]
    assert events[-1]["mode"] == "guided"
    assert client.get("/api/report/101/2019-07-31").json()["actuation"] is False
    assert (
        client.post(
            "/api/chat", json={"home": "999", "date": "2019-07-31", "message": "hi"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/chat", json={"home": "../101", "date": "2019-07-31", "message": "hi"}
        ).status_code
        == 422
    )


def test_llm_selection_is_bounded_and_cannot_override_scope(monkeypatch):
    monkeypatch.setenv("ENERGY_LLM_URL", "http://127.0.0.1:11434/api/chat")
    monkeypatch.setenv("ENERGY_LLM_MODEL", "test")
    response = {
        "message": {"tool_calls": [{"function": {"name": "compare_schedules", "arguments": {}}}]}
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps(response).encode())
    )
    assert llm_tools("reduce my bill") == ["compare_schedules"]
    response["message"]["tool_calls"][0]["function"]["arguments"] = {"home": "999"}
    with pytest.raises(ValueError, match="override"):
        llm_tools("reduce my bill")


def test_provider_failure_is_visible_and_upgrade_claims_are_refused(bundle, monkeypatch):
    monkeypatch.setattr(
        "energy_assistant.agent.llm_tools", lambda *a: (_ for _ in ()).throw(OSError("offline"))
    )
    result = list(
        chat_events(
            EvidenceService(bundle), "Would insulation reduce my bill?", "101", "2019-07-31"
        )
    )[-1]
    assert result["mode"] == "guided"
    assert result["notice"]
    assert "cannot estimate" in result["text"]


def test_no_side_effect_from_comparison(bundle):
    service = EvidenceService(bundle)
    before = deepcopy(service.data)
    service.compare_schedules("101", "2019-07-31")
    assert service.data == before


def test_live_forecast_hides_future_and_reuses_fitted_context(bundle, monkeypatch):
    from types import SimpleNamespace

    data = json.loads((bundle / "evidence.json").read_text())
    data["model"].update(backend="tabpfn", context_rows=32)
    record = data["homes"]["101"]
    record["dates"]["2019-07-31"]["forecast"]["rows"] = [{"hour": hour} for hour in range(12, 24)]
    days = np.ones((1, 24, 4))
    days[:, 13:] = 999999  # Held-out future must never reach a live query.
    path = bundle / "home_101_context.npz"
    np.savez(path, train=np.zeros((2, 24, 4)), days=days, dates=["2019-07-31"])
    record["live_context_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_bundle(bundle, data)
    fits, histories = [], []

    def fit(train, context):
        fits.append(train.copy())

        def predict(history):
            histories.append(history.copy())
            return np.ones((12, 4))

        predictor = SimpleNamespace(
            predict_prefix=predict, models=[SimpleNamespace(predict=lambda rows: rows[:, 0])]
        )
        return predictor, np.zeros((8, 16))

    monkeypatch.setattr("energy_assistant.prepare.fit_tabpfn", fit)
    service = EvidenceService(bundle, live=True)
    first = service.forecast_and_explain("101", "2019-07-31")
    second = service.forecast_and_explain("101", "2019-07-31")
    assert len(fits) == 1
    assert all(h.shape == (13, 4) and np.max(h) == 1 for h in histories)
    assert not first["runtime"]["reused_fitted_context"]
    assert second["runtime"]["reused_fitted_context"]


def test_overview_uses_same_plan_for_cash_and_comfort(bundle):
    service = EvidenceService(bundle)
    day = service.data["homes"]["101"]["dates"]["2019-07-31"]
    day["scenarios"]["seasonal"].update(bill=-2.25, degree_hours=12, comfort_pct=75)
    overview = service.overview("101")
    result = next(p for p in overview["plans"] if p["id"] == "seasonal")["days"][0]
    assert result["net_cash"] == 2.25  # Credit to the household, not a payment.
    assert result["comfort_band"] == 5
    assert result["comfort_pct"] == 75
    assert overview["evidence"] == "simulated"
    baseline = next(p for p in overview["plans"] if p["id"] == "feedback")["days"][0]
    assert baseline["net_cash"] == -9.84
    assert baseline["comfort_band"] == 0


def test_period_totals_gaps_and_daily_guards(bundle):
    from energy_assistant.periods import PeriodService
    from energy_assistant.presentation import tool_response

    service = EvidenceService(bundle)
    record = service.data["homes"]["101"]
    record["dates"]["2019-08-02"] = deepcopy(record["dates"]["2019-07-31"])
    record["bill"]["days"].append(deepcopy(record["bill"]["days"][0]))
    record["bill"]["days"][-1]["date"] = "2019-08-02"
    scope = PeriodService(service, "2019-07-31", "2019-08-02")
    bill = tool_response(scope, "explain_bill", "101", "2019-07-31")
    assert bill["costs"]["period_total"] == pytest.approx(19.68)
    assert "most_expensive_hour" not in bill["costs"]
    assert "day_total" not in bill["costs"]
    assert bill["selected_period"]["available_days"] == 2
    assert bill["selected_period"]["requested_days"] == 3
    assert not bill["selected_period"]["complete"]
    assert scope.call("compare_schedules", "101", "2019-07-31")["recommended"] == "seasonal"
    # One failed service day cannot be averaged away by an otherwise cheaper month.
    record["dates"]["2019-08-02"]["scenarios"]["seasonal"]["ev_complete"] = False
    result = scope.call("compare_schedules", "101", "2019-07-31")
    assert result["recommended"] is None
    assert result["candidates"][1]["days_failing_checks"] == 1
    assert result["candidates"][1]["bill"] == 16
    assert "Missing days" in bill["context"]


def test_single_home_binding_excludes_other_households(bundle):
    data = json.loads((bundle / "evidence.json").read_text())
    data["homes"]["102"] = deepcopy(data["homes"]["101"])
    write_bundle(bundle, data)
    with pytest.raises(ValueError, match="--home"):
        create_app(bundle)
    client = TestClient(create_app(bundle, home=101))
    assert [h["id"] for h in client.get("/api/catalog").json()["homes"]] == ["101"]
    assert client.get("/api/overview").json()["home"] == "101"
    assert client.get("/api/report/102/2019-07-31").status_code == 404


def test_settings_key_is_private_and_changes_need_session_token(bundle):
    client = TestClient(create_app(bundle))
    body = {"provider": "openai", "model": "test-model", "api_key": "test-private-value"}
    assert client.post("/api/settings", json=body).status_code == 403
    token = client.get("/api/catalog").json()["settings_token"]
    response = client.post("/api/settings", json=body, headers={"X-Settings-Token": token})
    assert response.status_code == 200
    assert response.json()["has_key"]
    assert "test-private-value" not in response.text
    assert "api_key" not in client.get("/api/settings").json()
    assert (bundle / "chat_settings_101.json").stat().st_mode & 0o777 == 0o600
    response = client.post(
        "/api/settings",
        json={"provider": "anthropic", "model": "test-claude"},
        headers={"X-Settings-Token": token},
    )
    assert not response.json()["has_key"]  # Never carry an OpenAI key to another provider.
    assert response.json()["endpoint"] == "https://api.anthropic.com/v1/messages"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_native_hosted_tool_protocols(monkeypatch, provider):
    captured = {}

    def reply(request, **kwargs):
        captured["body"] = json.loads(request.data)
        captured["headers"] = dict(request.header_items())
        response = (
            {"output": [{"type": "function_call", "name": "explain_bill", "arguments": "{}"}]}
            if provider == "openai"
            else {"content": [{"type": "tool_use", "name": "explain_bill", "input": {}}]}
        )
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("urllib.request.urlopen", reply)
    config = {
        "provider": provider,
        "model": "test-model",
        "endpoint": "https://api.openai.com/v1/responses"
        if provider == "openai"
        else "https://api.anthropic.com/v1/messages",
        "api_key": "private",
    }
    assert llm_tools("Why did this cost so much?", config=config) == ["explain_bill"]
    if provider == "openai":
        assert captured["body"]["store"] is False
        assert all(
            t["strict"] and t["parameters"]["required"] == [] for t in captured["body"]["tools"]
        )
        assert captured["headers"]["Authorization"] == "Bearer private"
    else:
        assert captured["headers"]["X-api-key"] == "private"
        assert "Authorization" not in captured["headers"]
        assert all("input_schema" in t for t in captured["body"]["tools"])
        assert captured["body"]["tool_choice"] == {"type": "auto"}


def test_launcher_resolves_supported_models_and_provider_keys(monkeypatch):
    from hems_assistant import chat_configuration

    monkeypatch.setenv("OPEN_API_KEY", "openai-only")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert (
        chat_configuration("qwen35_2b", 9881)["endpoint"]
        == "http://127.0.0.1:9881/v1/chat/completions"
    )
    assert chat_configuration("openai:gpt-6", 9881)["api_key"] == "openai-only"
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        chat_configuration("anthropic:test-claude", 9881)
    with pytest.raises(ValueError, match="Use qwen35_2b"):
        chat_configuration("unsupported_local_model", 9881)
    with pytest.raises(ValueError, match="HTTPS"):
        chat_configuration(
            "compatible:test", 9881, endpoint="http://remote.example/v1/chat/completions"
        )


def test_local_cpu_budget_and_occupied_port(monkeypatch):
    import socket

    from energy_assistant.local_llm import cpu_threads, start

    monkeypatch.setattr("os.sched_getaffinity", lambda _: set(range(2)))
    assert cpu_threads() == 1
    assert cpu_threads(2) == 2
    with pytest.raises(ValueError):
        cpu_threads(3)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        with pytest.raises(ValueError, match="already in use"):
            start(listener.getsockname()[1])


def test_appliance_costs_reconcile_without_double_counting_solar(bundle):
    from energy_assistant.insights import breakdown

    service = EvidenceService(bundle)
    record = service.data["homes"]["101"]
    hour = {
        "hour": 0,
        "devices_kwh": {"everyday": 1.0, "cooling": 2.0, "car": 0.0, "laundry": 1.0},
        "devices_gross_cost": {"everyday": 0.2, "cooling": 0.4, "car": 0.0, "laundry": 0.2},
        "solar_used_credit": 0.6,
        "export_credit": 0.1,
        "fixed_cost": 0.05,
        "net_cost": 0.15,
    }
    record["bill"]["days"] = [{"date": "2019-07-31", "hours": [hour], "total": 0.15}]
    result = breakdown(service, "101", ["2019-07-31"])
    assert sum(r["gross_cost"] for r in result["devices"]) == pytest.approx(0.8)
    assert sum(r["cost"] for r in result["adjustments"]) == pytest.approx(-0.65)
    assert result["total"] == pytest.approx(0.15)
    record["bill"]["days"][0]["total"] = 10
    with pytest.raises(ValueError, match="reconcile"):
        breakdown(service, "101", ["2019-07-31"])


def test_selected_plan_is_comparison_reference(bundle):
    from energy_assistant.insights import schedule_comparison

    service = EvidenceService(bundle)
    result = schedule_comparison(service, "101", ["2019-07-31"], "seasonal")
    assert not result["improvement_found"]
    assert result["savings"] == 0
    assert result["reference"]["id"] == "seasonal"
    result = schedule_comparison(service, "101", ["2019-07-31"], "feedback")
    assert result["improvement_found"]
    assert result["savings"] == pytest.approx(1.84)


def test_forecast_planning_does_not_read_actual_future(tmp_path):
    from types import SimpleNamespace

    import pandas as pd

    from dataset import _construct_dataset
    from demo import generate_data
    from em_strategy import apply_em_strategy, compose_em_strategy, make_em_strategy
    from energy_assistant.simulation import forecast_outlook
    from environment import HOME_ENERGY_MGNT

    data = tmp_path / "generated"
    generate_data(data)
    train, heldout, dates, scaler = _construct_dataset(
        pd.read_csv(data / "split_homes_clean/home_101.csv"),
        temp_price_path=data / "temp_price_newyork.csv",
        validation_days=14,
        split="validation",
    )
    client = SimpleNamespace(train_data=train, scaler=scaler)
    strategy = compose_em_strategy(make_em_strategy(), [client])
    env = HOME_ENERGY_MGNT(heldout[0], scaler=scaler, state_dim=17)
    apply_em_strategy(env, strategy)
    for _ in range(6):
        env.step((0, np.array([1.0, 1.0, 0.0])))
    predicted = np.tile([1.0, 0.5, 25.0, 0.2], (24, 1))
    first = forecast_outlook(env, client, strategy, predicted, dates[0], 5.0)
    env.dataset = env.dataset.copy()
    env.dataset[6:, [0, 1, 3, 4, 5, 6, 7]] = 999999
    second = forecast_outlook(env, client, strategy, predicted, dates[0], 5.0)
    assert first == second
    assert sum(r["net_cost"] for r in first["ledger"]) == pytest.approx(first["remaining_cost"])
    assert len(first["actions"]) == 18
    assert all(0 <= r["battery_soc"] <= 1 for r in first["actions"])


@pytest.mark.parametrize(
    "question,tool",
    [
        ("Give me a breakdown of my electricity costs", "appliance_breakdown"),
        ("Which appliances caused this bill?", "appliance_breakdown"),
        ("What could I have done differently?", "schedule_comparison"),
        ("When should I turn on the AC today?", "plan_day"),
        ("When should I charge my car?", "plan_day"),
    ],
)
def test_everyday_question_fallback(question, tool):
    from energy_assistant.agent import guided_tools

    assert guided_tools(question) == [tool]


def test_history_browsing_never_fabricates_model_results(bundle):
    from energy_assistant.periods import PeriodService
    from energy_assistant.presentation import tool_response

    service = EvidenceService(bundle)
    record = service.data["homes"]["101"]
    record["history_bill"] = deepcopy(record["bill"])
    earlier = deepcopy(record["bill"]["days"][0])
    earlier["date"] = "2019-05-01"
    record["history_bill"]["days"].insert(0, earlier)
    record["history_phase"] = {"2019-05-01": "Training history", "2019-07-31": "Validation period"}
    assert service.catalog()["homes"][0]["dates"] == ["2019-05-01", "2019-07-31"]
    assert service.catalog()["homes"][0]["model_dates"] == ["2019-07-31"]
    scope = PeriodService(service, "2019-05-01", "2019-05-31")
    assert scope.call("explain_bill", "101", "2019-05-01")["total"] == pytest.approx(9.84)
    result = scope.call("forecast_and_explain", "101", "2019-05-01")
    assert result["available"] is False
    assert "Training history" in result["note"]
    response = tool_response(scope, "forecast_and_explain", "101", "2019-05-01")
    assert "forecast" not in response
    assert "no prepared model replay" in response["answer"]
    assert (
        next(p for p in service.overview("101")["plans"] if p["id"] == "recorded")["days"][0][
            "comfort_pct"
        ]
        is None
    )


def test_trained_forecaster_caps_queries_at_training_horizon(monkeypatch):
    from energy_assistant.trained_forecast import TrainedForecaster
    from forecasting import DirectForecaster

    seen = []

    def predict(self, x, current, origin, target):
        seen.append((x.copy(), target.copy()))
        return np.ones((len(x), 4))

    monkeypatch.setattr(DirectForecaster, "_predict", predict)
    predictor = TrainedForecaster("tabpfn")
    predictor.horizon = 6
    x = np.zeros((2, 16))
    current = np.ones((2, 4))
    origin = np.array([6, 6])
    target = np.array([9, 21])
    predictor._predict(x, current, origin, target)
    bounded, target = seen[0]
    assert target.tolist() == [9, 12]
    assert bounded[1, 13] == 6 / 24
    assert bounded[1, 15] == pytest.approx(-1)
    assert not x.any()  # Caller inputs remain unchanged.


def test_standalone_question_does_not_reexecute_history(monkeypatch):
    requests = []

    def response(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "forecast_and_explain",
                                            "arguments": "{}",
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", response)
    config = {
        "provider": "local",
        "model": "test",
        "endpoint": "http://localhost/v1/chat/completions",
    }
    llm_tools("Explain the TabPFN forecast", history=["Which appliances cost most?"], config=config)
    assert len(requests[-1]["messages"]) == 2
    assert "appliances cost most" not in requests[-1]["messages"][-1]["content"]
    llm_tools("And what about the car?", history=["When should I run laundry?"], config=config)
    assert "When should I run laundry?" in requests[-1]["messages"][-1]["content"]


@pytest.mark.parametrize(
    "variable,column,unit", [("demand", 0, "kWh"), ("solar", 1, "kWh"), ("temperature", 2, "°C")]
)
def test_specific_forecast_explanation_is_causal_and_unit_correct(
    bundle, monkeypatch, variable, column, unit
):
    from types import SimpleNamespace

    data = json.loads((bundle / "evidence.json").read_text())
    data["model"].update(backend="tabpfn", context_rows=32)
    record = data["homes"]["101"]
    record["forecast_context"] = {"horizon": 6}
    days = np.tile([2.0, 1.0, -5.0, 0.2], (1, 24, 1))
    days[:, 1:] = 999999  # Origin zero must never read these rows.
    path = bundle / "home_101_context.npz"
    np.savez(path, train=np.zeros((2, 24, 4)), days=days, dates=["2019-07-31"])
    record["live_context_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_bundle(bundle, data)
    calls = []

    def fit(train, context):
        calls.append(1)
        return SimpleNamespace(
            models=[SimpleNamespace(predict=lambda x, c=c: x[:, c]) for c in range(3)]
        ), np.zeros((8, 16))

    monkeypatch.setattr("energy_assistant.prepare.fit_tabpfn", fit)
    app = create_app(bundle, live=True)
    client = TestClient(app)
    url = f"/api/explanation/101/2019-07-31?origin=0&target=3&variable={variable}"
    response = client.get(url)
    assert response.status_code == 200
    result = response.json()
    assert result["unit"] == unit
    assert result["prediction"] == pytest.approx([2, 1, -5][column])
    assert result["baseline"] + sum(r["value"] for r in result["factors"]) == pytest.approx(
        result["prediction"]
    )
    assert result["recent_readings"] == [{"hour": 0, "value": [2, 1, -5][column]}]
    assert max(r["hour"] for r in result["observed_inputs"]) == 0
    assert client.get(url).json() == result
    assert len(calls) == 1
    for query in ("origin=0&target=7", "origin=6&target=6", "origin=-1&target=3"):
        assert client.get("/api/explanation/101/2019-07-31?" + query).status_code == 400
    assert client.get("/api/explanation/950/2019-07-31").status_code == 400


def test_peer_trading_question_does_not_invent_market(bundle, monkeypatch):
    from energy_assistant.periods import PeriodService

    monkeypatch.setattr("energy_assistant.agent.llm_tools", lambda *a: ["inspect_home"])
    result = list(
        chat_events(
            PeriodService(EvidenceService(bundle), "2019-07-31", None),
            "When should I sell energy to a neighbour?",
            "101",
            "2019-07-31",
        )
    )[-1]
    assert result["event"] == "answer"
    assert "Neighbour-to-neighbour trading is not enabled" in result["text"]
    assert [c["tool"] for c in result["cards"]] == ["appliance_breakdown", "plan_day"]


def test_exchange_unavailable_without_metered_flows(bundle):
    from energy_assistant.insights import breakdown

    result = breakdown(EvidenceService(bundle), "101", ["2019-07-31"])
    assert result["available"] is False


@pytest.mark.parametrize("hour", [0, 6, 12, 18])
def test_report_accepts_browser_query_hour_and_downloads(bundle, hour):
    client = TestClient(create_app(bundle))
    response = client.get(
        f"/api/report/101/2019-07-31?end_date=2019-07-31&hour={hour}&plan=feedback"
    )
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment;")
    assert response.json()["planning_hour"] == hour
    assert client.get("/api/report/101/2019-07-31?hour=7").status_code == 404


def test_printable_report_is_downloadable_and_escapes_input(bundle):
    from energy_assistant.periods import PeriodService
    from energy_assistant.report import render_report

    client = TestClient(create_app(bundle))
    response = client.get("/api/report/101/2019-07-31?hour=6&format=html")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '.html"' in response.headers["content-disposition"]
    assert "Your energy report" in response.text and "$9.84" in response.text
    report = PeriodService(EvidenceService(bundle), "2019-07-31", None).call(
        "export_plan", "101", "2019-07-31"
    )
    report["home"] = "<script>alert(1)</script>"
    assert "<script>" not in render_report(report)
    assert "&lt;script&gt;" in render_report(report)


@pytest.mark.parametrize("tampered", [False, True])
def test_launcher_verifies_new_home_bundle_and_preserves_test_split(
    tmp_path, monkeypatch, tampered
):
    from hems_assistant import resolve_model

    descriptor = {
        "bundle_type": "local_reference",
        "home_id": 27,
        "run_dir": str(tmp_path / "completed_run"),
        "checkpoint_name": "latest",
        "assistant_contract": {"split": "test"},
    }
    (tmp_path / "home_model.json").write_text(json.dumps(descriptor))
    verified = []

    def load(directory):
        verified.append(directory)
        if tampered:
            raise ValueError("Home model manifest changed")
        return descriptor

    monkeypatch.setattr("gridpfn.deployment.load_model_bundle", load)
    monkeypatch.setattr(
        "utils.run_io.read_logged_settings",
        lambda path: {"home_ids": [27], "data_period": "month_10_refit"},
    )
    if tampered:
        with pytest.raises(ValueError, match="manifest changed"):
            resolve_model(tmp_path)
    else:
        run, home, checkpoint, settings = resolve_model(tmp_path)
        assert (run, home, checkpoint) == (tmp_path / "completed_run", 27, "latest")
        assert settings["assistant_split"] == "test"
        assert settings["data_period"] == "month_10_refit"
    assert verified == [tmp_path]


def test_full_recorded_history_preserves_training_scale_and_missing_dates(tmp_path):
    import pandas as pd

    from dataset import _construct_dataset
    from demo import generate_data
    from energy_assistant.history import recorded_history_bundle

    generate_data(tmp_path)
    home = tmp_path / 'split_homes_clean/home_101.csv'
    weather = tmp_path / 'temp_price_newyork.csv'
    frame = pd.read_csv(home)
    frame = frame[~frame.datetime.str.startswith('2019-08-15')]
    frame.to_csv(home, index=False)
    bundle = _construct_dataset(frame, temp_price_path=weather, validation_days=14)
    expanded = recorded_history_bundle(bundle, home, weather)
    assert expanded[0] is bundle[0] and expanded[3] is bundle[3]
    assert len(expanded[2]) == 91
    assert (expanded[2][0], expanded[2][-1]) == ('2019-06-01', '2019-08-31')
    assert '2019-08-15' not in expanded[2]
    assert np.isfinite(expanded[1]).all()
    training_indices = [expanded[2].index(day) for day in bundle[3]['train_dates']]
    np.testing.assert_allclose(expanded[1][training_indices], bundle[0])


@pytest.mark.parametrize('option', ['--household', '--model'])
@pytest.mark.parametrize('restricted', [False, True])
def test_household_launcher_prepares_once_and_binds_full_history(
    tmp_path, monkeypatch, option, restricted
):
    import uvicorn

    import energy_assistant.local_llm as local_llm
    import energy_assistant.prepare as preparation
    import energy_assistant.web as web
    import hems_assistant

    settings = {'home_ids': [27], 'path_data': str(tmp_path / 'data'), 'assistant_split': 'test'}
    monkeypatch.setattr(hems_assistant, 'resolve_model', lambda *a: (tmp_path, 27, 'latest', settings))
    monkeypatch.setattr(hems_assistant, 'evidence_cache_key', lambda *a, **kw: 'cache')
    monkeypatch.setattr(preparation, 'ROOT', tmp_path)
    monkeypatch.setattr(preparation, 'source_hashes', lambda: {})
    prepared, served, downloads = [], [], []

    def prepare(path, **kwargs):
        prepared.append(kwargs)
        path.mkdir(parents=True)
        (path / 'evidence.json').write_text('{}')

    monkeypatch.setattr(preparation, 'prepare', prepare)
    monkeypatch.setattr(local_llm, 'setup', lambda: downloads.append(True))
    monkeypatch.setattr(web, 'create_app', lambda path, **kw: served.append(kw) or object())
    monkeypatch.setattr(uvicorn, 'run', lambda *a, **kw: None)
    args = [option, str(tmp_path / 'home_27'), '--llm', 'qwen35_2b']
    if restricted:
        args += ['--split', 'test', '--days', '3']
    hems_assistant.main(args)
    hems_assistant.main(args)
    assert len(prepared) == 1  # reopening reuses the completed prepared household
    assert prepared[0]['history'] is (not restricted)
    assert prepared[0]['days'] == (3 if restricted else None)
    assert prepared[0]['split'] == 'test' and prepared[0]['homes'] == [27]
    assert all(s['home'] == 27 and s['live'] and s['local_chat'] for s in served)
    assert downloads == [True, True]
