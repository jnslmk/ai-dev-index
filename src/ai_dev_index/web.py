from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ai_dev_index.config import Settings
from ai_dev_index.db import Database
from ai_dev_index.pipeline import SyncPipeline
from ai_dev_index.scheduler import run_periodic_sync

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"


def _format_date(iso_str: str | None) -> str:
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d, %Y")
    except ValueError:
        return iso_str[:10]


def _format_ts(seconds: float | None) -> str:
    sec = max(0, int(seconds or 0))
    hours, rem = divmod(sec, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    db = Database(cfg.db_path)
    pipeline = SyncPipeline(db, cfg)
    pipeline.load_configured_channels()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        scheduler_task: asyncio.Task[None] | None = None
        if cfg.enable_scheduler:
            scheduler_task = asyncio.create_task(run_periodic_sync(pipeline, cfg))
        try:
            yield
        finally:
            if scheduler_task is not None:
                scheduler_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await scheduler_task

    app = FastAPI(
        title="ai-dev-index",
        description="Self-hosted index, summaries, and trend radar for AI engineering YouTube channels",
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.db = db
    app.state.pipeline = pipeline

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["format_date"] = _format_date
    templates.env.filters["format_ts"] = _format_ts

    def _base_ctx(request: Request, active_nav: str) -> dict[str, Any]:
        return {
            "request": request,
            "active_nav": active_nav,
            "stats": db.get_stats(),
            "llm_enabled": cfg.llm_enabled,
            "llm_model": cfg.llm_model,
            "sync_running": pipeline.is_running,
        }

    @app.get("/", response_class=HTMLResponse)
    async def home(
        request: Request,
        q: str = Query(default=""),
        channel: str = Query(default=""),
        category: str = Query(default=""),
        entity: str = Query(default=""),
        days: int = Query(default=30, ge=0, le=3650),
    ) -> HTMLResponse:
        videos = db.list_videos(
            q=q,
            channel_id=channel,
            category=category,
            entity=entity,
            days=days if days > 0 else None,
            limit=60,
        )
        trends = db.get_trends(days=14, limit=16)
        channels = db.list_channels_with_counts()
        categories = sorted({c["category"] for c in channels})
        ctx = {
            **_base_ctx(request, "feed"),
            "videos": videos,
            "trends": trends,
            "channels": channels,
            "categories": categories,
            "filters": {
                "q": q,
                "channel": channel,
                "category": category,
                "entity": entity,
                "days": days,
            },
        }
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.get("/trends", response_class=HTMLResponse)
    async def trends_page(
        request: Request,
        days: int = Query(default=14, ge=1, le=365),
        topic: str = Query(default=""),
    ) -> HTMLResponse:
        trends = db.get_trends(days=days, limit=40)
        selected_topic = topic or (trends[0]["name"] if trends else "")
        topic_videos = (
            db.list_videos(entity=selected_topic, days=days, limit=30) if selected_topic else []
        )
        max_videos = max((t["video_count"] for t in trends), default=1)
        ctx = {
            **_base_ctx(request, "trends"),
            "days": days,
            "trends": trends,
            "max_videos": max_videos,
            "selected_topic": selected_topic,
            "topic_videos": topic_videos,
        }
        return templates.TemplateResponse(request, "trends.html", ctx)

    @app.get("/channels", response_class=HTMLResponse)
    async def channels_page(request: Request) -> HTMLResponse:
        channels = db.list_channels_with_counts()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for ch in channels:
            grouped.setdefault(ch["category"], []).append(ch)
        ctx = {
            **_base_ctx(request, "channels"),
            "grouped_channels": grouped,
        }
        return templates.TemplateResponse(request, "channels.html", ctx)

    @app.get("/video/{video_id}", response_class=HTMLResponse)
    async def video_detail(request: Request, video_id: str) -> HTMLResponse:
        video = db.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        ctx = {
            **_base_ctx(request, "feed"),
            "video": video,
            "store_transcripts": cfg.store_transcripts,
        }
        return templates.TemplateResponse(request, "video.html", ctx)

    @app.get("/api/health")
    async def api_health() -> dict[str, Any]:
        return {
            "status": "ok",
            "sync_running": pipeline.is_running,
            "llm_enabled": cfg.llm_enabled,
        }

    @app.get("/api/stats")
    async def api_stats() -> dict[str, Any]:
        return db.get_stats()

    @app.get("/api/videos")
    async def api_videos(
        q: str = Query(default=""),
        channel: str = Query(default=""),
        category: str = Query(default=""),
        entity: str = Query(default=""),
        days: int = Query(default=30, ge=0, le=3650),
        limit: int = Query(default=40, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        videos = db.list_videos(
            q=q,
            channel_id=channel,
            category=category,
            entity=entity,
            days=days if days > 0 else None,
            limit=limit,
            offset=offset,
        )
        return {"count": len(videos), "videos": videos}

    @app.get("/api/videos/{video_id}")
    async def api_video_detail(video_id: str) -> dict[str, Any]:
        video = db.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        return video

    @app.get("/api/trends")
    async def api_trends(
        days: int = Query(default=14, ge=1, le=365),
        limit: int = Query(default=25, ge=1, le=100),
    ) -> dict[str, Any]:
        return {"days": days, "trends": db.get_trends(days=days, limit=limit)}

    @app.post("/api/sync")
    async def api_trigger_sync(
        background_tasks: BackgroundTasks,
        limit_per_channel: int | None = Query(default=None, ge=1, le=15),
        reanalyze_heuristic: bool = Query(default=False),
    ) -> JSONResponse:
        if pipeline.is_running:
            return JSONResponse(
                status_code=409,
                content={"status": "already_running", "message": "Sync is already running"},
            )

        def _run() -> None:
            try:
                pipeline.sync(
                    limit_per_channel=limit_per_channel,
                    reanalyze_heuristic=reanalyze_heuristic,
                )
            except Exception:
                logger.exception("Manual sync failed")

        background_tasks.add_task(_run)
        return JSONResponse(
            status_code=202,
            content={"status": "started", "message": "Background sync started"},
        )

    @app.get("/feed.xml")
    async def atom_feed(request: Request) -> Response:
        videos = db.list_videos(limit=30)
        base_url = str(request.base_url).rstrip("/")
        updated = videos[0]["published_at"] if videos else ""
        entries = []
        for v in videos:
            takeaways_html = "".join(f"<li>{escape(t)}</li>" for t in v["takeaways"])
            content = f"<p>{escape(v['summary'])}</p>"
            if takeaways_html:
                content += f"<ul>{takeaways_html}</ul>"
            content += f'<p><a href="{escape(v["url"])}">Watch on YouTube</a></p>'
            entries.append(
                f"""  <entry>
    <id>yt:video:{escape(v['video_id'])}</id>
    <title>{escape(v['channel_name'])}: {escape(v['title'])}</title>
    <link href="{escape(base_url + '/video/' + v['video_id'])}" />
    <published>{escape(v['published_at'])}</published>
    <updated>{escape(v['updated_at'])}</updated>
    <author><name>{escape(v['channel_name'])}</name></author>
    <summary type="html">{escape(content)}</summary>
  </entry>"""
            )

        xml = f"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ai-dev-index</title>
  <id>{escape(base_url)}/feed.xml</id>
  <link href="{escape(base_url)}/feed.xml" rel="self" />
  <link href="{escape(base_url)}/" />
  <updated>{escape(updated)}</updated>
{chr(10).join(entries)}
</feed>
"""
        return Response(content=xml, media_type="application/atom+xml; charset=utf-8")

    return app
