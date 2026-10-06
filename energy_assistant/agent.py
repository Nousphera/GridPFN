"""Optional LLM tool selection with bounded, grounded tool execution.

With no provider, clearly labeled guided mode remains runnable. The LLM chooses
tools; it never supplies the final financial numbers or device instructions.
"""

import json
import os
import re
import urllib.request

from .presentation import presentation
from .service import TOOL_NAMES, narrative

DESCRIPTIONS = {
    "inspect_home": "Learn what we know about this home and which questions we can answer.",
    "explain_bill": "See where the electricity costs come from and which hours cost the most.",
    "forecast_and_explain": "Explain TabPFN's everyday-demand forecast and its feature influences. This tool does NOT project battery charge or appliance schedules; use plan_day for those.",
    "compare_schedules": "Find out whether a different energy plan could cost less while meeting the home's needs.",
    "export_plan": "Get a report with the costs, forecast and energy plans we compared.",
    "appliance_breakdown": "Break down recorded electricity costs by appliance: cooling, car charging, laundry and everyday use, with solar credits. Use for which devices cost most or a bill breakdown.",
    "schedule_comparison": "Show side-by-side appliance timelines for the selected energy plan and a cheaper tested alternative, priced using observed tariffs. Use for what could I have done differently, ideal schedule or how to reduce a bill retrospectively. Neither plan is claimed globally optimal.",
    "plan_day": "Plan when to cool, charge the car or battery, or run laundry using a dated TabPFN forecast from the selected planning time. Includes solar, temperature, assumed prices and projected battery charge. Use for when should I or looking ahead. This is not a live device connection or a true price forecast.",
}

ACTIVITY = {
    "inspect_home": "Getting to know this home…",
    "explain_bill": "Looking at when electricity cost the most…",
    "forecast_and_explain": "Checking the forecast and what shaped it…",
    "compare_schedules": "Comparing costs, comfort, charging and laundry…",
    "export_plan": "Putting your home energy report together…",
    "appliance_breakdown": "Adding up appliance use and solar credits…",
    "schedule_comparison": "Comparing your plan, hour by hour…",
    "plan_day": "Looking ahead at solar, comfort and appliance needs…",
}


def guided_tools(message):
    message = message.lower()
    if any(
        word in message for word in ("insulation", "heat pump", "retrofit", "new solar", "upgrade")
    ):
        return ["inspect_home"]
    if any(word in message for word in ("download", "export", "report")):
        return ["export_plan"]
    if any(
        word in message
        for word in (
            "breakdown",
            "break down",
            "which device",
            "which appliance",
            "caused",
            "used the most",
        )
    ):
        return ["appliance_breakdown"]
    if any(
        word in message
        for word in (
            "when should",
            "when to",
            "turn on",
            "do the laundry",
            "solar",
            "battery state",
            "battery charge",
            "price forecast",
        )
    ):
        return ["plan_day"]
    if any(
        word in message
        for word in (
            "could i",
            "could have",
            "side by side",
            "side-by-side",
            "ideal",
            "differently",
        )
    ):
        return ["schedule_comparison"]
    result = []
    wants_comparison = any(
        word in message
        for word in ("reduce", "save", "saving", "schedule", "compare", "charge", "cheaper")
    )
    if "why" in message or (
        not wants_comparison and any(word in message for word in ("bill", "cost", "expensive"))
    ):
        result.append("explain_bill")
    if any(word in message for word in ("forecast", "predict", "shap", "influence", "interpret")):
        result.append("forecast_and_explain")
    if wants_comparison:
        result.append("compare_schedules")
    return result or ["inspect_home"]


def llm_tools(message, history=(), config=None):
    """Ollama or an OpenAI-compatible tool-calling endpoint, explicitly configured."""
    if config and config.get("provider") == "guided":
        return None
    endpoint = config["endpoint"] if config else os.environ.get("ENERGY_LLM_URL", "")
    model = config["model"] if config else os.environ.get("ENERGY_LLM_MODEL", "")
    if not endpoint or not model:
        return None
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": DESCRIPTIONS[name],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }
        for name in TOOL_NAMES
    ]
    messages = [
        {
            "role": "system",
            "content": "You are the tool planner for a home-energy research demo. Select 1 to 3 tools for "
            "the user's question. Home and date are bound by the application. Call tools with "
            "empty arguments. Never claim to actuate devices, diagnose equipment, or estimate "
            "retrofit ROI. For unsupported requests use inspect_home. Do not invent numbers. "
            "Forecast explanations describe an estimate, not causes. Schedule results are tests on past days. "
            "Interpret the final question as the current request; earlier questions only clarify follow-ups such as 'why then?' or 'what about the car?'. Battery state or charge projections REQUIRE plan_day, including combined battery and solar questions. For appliance timing use plan_day; for a bill breakdown use appliance_breakdown; for retrospective changes use schedule_comparison. Use plain, friendly language. Do not describe caches, execution logs or model internals "
            "unless asked. Keep the distinction between example data, forecasts and simulated savings clear.",
        }
    ]
    followup = re.search(
        r"^(and\b|what about\b|why then\b|why that\b|how about\b|can you explain that\b)",
        message.strip(),
        re.I,
    )
    current = message
    if followup and history:
        current = (
            "Earlier questions (context only):\n"
            + "\n".join(str(item)[:700] for item in history[-2:])
            + "\n\nAnswer only this current request:\n"
            + message
        )
    messages.append({"role": "user", "content": current})
    payload = {"model": model, "messages": messages, "tools": tools, "stream": False}
    ollama = endpoint.rstrip("/").endswith("/api/chat")
    responses = endpoint.rstrip("/").endswith("/responses")
    anthropic = config is not None and config.get("provider") == "anthropic"
    if anthropic:
        payload = {
            "model": model,
            "system": messages[0]["content"],
            "messages": messages[1:],
            "max_tokens": 1024,
            "tool_choice": {"type": "auto"},
            "tools": [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "input_schema": tool["function"]["parameters"],
                }
                for tool in tools
            ],
        }
    elif ollama:
        payload.update(think=False, options={"temperature": 0})
    elif responses:
        payload = {
            "model": model,
            "input": messages,
            "tool_choice": "required",
            "max_output_tokens": 1024,
            "store": False,
            "tools": [{"type": "function", **tool["function"], "strict": True} for tool in tools],
        }
    else:
        payload.update(tool_choice="required", temperature=0, max_tokens=256)
    headers = {"Content-Type": "application/json"}
    key = config.get("api_key") if config else os.environ.get("ENERGY_LLM_KEY")
    if anthropic:
        headers.update({"x-api-key": key or "", "anthropic-version": "2023-06-01"})
    elif key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(endpoint, json.dumps(payload).encode(), headers)
    with urllib.request.urlopen(request, timeout=45) as response:
        result = json.load(response)
    if anthropic:
        reply = {
            "tool_calls": [
                {"function": {"name": row["name"], "arguments": row["input"]}}
                for row in result["content"]
                if row.get("type") == "tool_use"
            ]
        }
    elif responses:
        reply = {
            "tool_calls": [
                {"function": {"name": row["name"], "arguments": row["arguments"]}}
                for row in result["output"]
                if row.get("type") == "function_call"
            ]
        }
    else:
        reply = result["message"] if ollama else result["choices"][0]["message"]
    chosen = []
    for call in reply.get("tool_calls", [])[:3]:
        function = call["function"]
        if function["name"] not in TOOL_NAMES:
            raise ValueError("Provider selected an unsupported tool")
        arguments = function.get("arguments", {})
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        if arguments != {}:
            raise ValueError("Tools cannot override the selected household or date")
        if function["name"] not in chosen:
            chosen.append(function["name"])
    if not chosen:
        raise ValueError("Provider did not return a tool call")
    # Demand explanations cannot answer battery-state questions. Enforce that
    # capability boundary even if the small planner selects a plausible wrong tool.
    if (
        re.search(r"battery", message, re.I)
        and re.search(r"\b(will|when|expect|predict|state|level)\b", message, re.I)
        and "plan_day" not in chosen
    ):
        chosen = [name for name in chosen if name != "forecast_and_explain"] + ["plan_day"]
    return chosen


def chat_events(service, message, home, date, history=(), config=None):
    service.select(home, date)
    yield {"event": "status", "text": "Looking at your question…"}
    mode = "guided"
    notice = None
    try:
        chosen = (
            llm_tools(message, history, config)
            if config is not None
            else llm_tools(message, history)
        )
        if chosen:
            mode = "llm"
    except (ValueError, KeyError, OSError, TimeoutError):
        chosen = None
        notice = (
            "The chat connection is unavailable, so we used guided mode to answer your question."
        )
    chosen = chosen or guided_tools(message)
    trading_question = bool(
        re.search(r"\b(sell|selling|trade|trading|neighbou?r|peer|surplus)\b", message, re.I)
    )
    if trading_question:
        chosen = ["appliance_breakdown", "plan_day"]
    cards, sentences = [], []
    for name in chosen:
        yield {"event": "tool_start", "tool": name, "text": ACTIVITY[name]}
        try:
            result = service.call(name, home, date)
        except (ValueError, OSError, RuntimeError, KeyError):
            yield {
                "event": "error",
                "text": "We couldn't check this result, so we can't give you a reliable answer yet. Please try again or ask the person who set up this demo for help.",
            }
            return
        cards.append({"tool": name, "data": result, "presentation": presentation(name, result)})
        sentences.append(narrative(name, result))
        yield {"event": "tool_done", "tool": name}
    if re.search(r"insulation|heat pump|retrofit|upgrade|new solar", message, re.I):
        sentences.insert(
            0,
            "We cannot estimate savings from that upgrade with the information available. We'd need details about your building, equipment and installation costs. We can still compare ways to schedule the appliances already included here.",
        )
    if trading_question:
        sentences.insert(
            0,
            "This home can receive grid-export credits in the simulation. Neighbour-to-neighbour trading is not enabled, so I cannot quote peer offers or promise the best selling time. Export credits below come from the recorded solar balance; the plan is a separate forecast-based simulation.",
        )
    yield {
        "event": "answer",
        "mode": mode,
        "notice": notice,
        "text": "\n\n".join(sentences),
        "cards": cards,
        "evidence_sha256": service.sha256,
    }
