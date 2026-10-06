"""Small local web app with a streamed, visible tool activity trace."""

import json
import secrets
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .agent import chat_events
from .periods import PeriodService
from .service import EvidenceService
from .settings import ProviderSettings, SettingsStore

STATIC = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    home: str = Field(pattern=r"^[0-9]{1,10}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    history: list[str] = Field(default_factory=list, max_length=4)
    plan: Literal["personalized", "feedback", "seasonal", "tabpfn"] | None = None
    hour: Literal[0, 6, 12, 18] = 6


def create_app(
    directory,
    live=False,
    local_chat=False,
    home=None,
    chat_port=8771,
    chat_config=None,
    chat_threads=None,
):
    service = EvidenceService(directory, live=live)
    home = service.bind_home(home)
    settings = SettingsStore(
        directory, local_enabled=local_chat, home=home, port=chat_port, initial=chat_config
    )
    settings_token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(_app):
        process = None
        if local_chat:
            from .local_llm import start

            process = start(chat_port, threads=chat_threads)
        try:
            yield
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    app = FastAPI(title="GridPFN Home", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"]
    )
    app.state.evidence = service
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/catalog")
    def catalog():
        return service.catalog() | {
            "agent_mode": "guided" if settings.public()["provider"] == "guided" else "llm",
            "chat_provider": settings.public()["provider"],
            "settings_token": settings_token,
        }

    @app.get("/api/settings")
    def read_settings():
        return settings.public()

    @app.get("/api/overview")
    def overview():
        return service.overview(home)

    @app.get("/api/explanation/{home}/{date}")
    def forecast_explanation(
        home: str,
        date: str,
        origin: int = 6,
        target: int = 12,
        variable: Literal["demand", "solar", "temperature"] = "demand",
    ):
        try:
            service.select(home, date)
            if service.live is None:
                return {
                    "available": False,
                    "note": "Forecast explanations require local TabPFN inference. Launch with live forecasts enabled.",
                }
            return service.live(
                home, date, origin_hour=origin, target_hour=target, target_name=variable
            )
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/settings")
    def save_settings(body: ProviderSettings, request: Request):
        if not secrets.compare_digest(request.headers.get("X-Settings-Token", ""), settings_token):
            raise HTTPException(403, "Refresh the application before changing settings")
        try:
            return settings.save(body)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/chat")
    def chat(body: ChatRequest):
        try:
            scope = PeriodService(service, body.date, body.end_date, body.plan, body.hour)
            scope.select(body.home, body.date)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

        def stream():
            for event in chat_events(
                scope, body.message, body.home, body.date, body.history, settings.private()
            ):
                yield json.dumps(event, allow_nan=False) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.get("/api/report/{home}/{date}")
    def report(
        home: str,
        date: str,
        end_date: str | None = None,
        plan: Literal["personalized", "feedback", "seasonal", "tabpfn"] | None = None,
        hour: int = 6,
        format: Literal["json", "html"] = "json",
    ):
        try:
            content = PeriodService(service, date, end_date, plan, hour).call(
                "export_plan", home, date
            )
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        from fastapi.responses import Response

        if format == "html":
            from .report import render_report

            return Response(
                render_report(content),
                media_type="text/html",
                headers={
                    "Content-Disposition": f'attachment; filename="gridpfn-home-{int(home)}-{date}.html"'
                },
            )
        return Response(
            json.dumps(content, indent=2, allow_nan=False),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="gridpfn-home-{int(home)}-{date}.json"'
            },
        )

    return app
