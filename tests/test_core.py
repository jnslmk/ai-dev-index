from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ai_dev_index.analyzer import analyze_heuristic, extract_known_entities
from ai_dev_index.config import Settings, load_channels
from ai_dev_index.db import Database
from ai_dev_index.ingest import parse_youtube_atom_feed
from ai_dev_index.web import create_app

SAMPLE_ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015"
      xmlns:media="http://search.yahoo.com/mrss/"
      xmlns="http://www.w3.org/2005/Atom">
  <title>Matt Pocock</title>
  <entry>
    <yt:videoId>abc123XYZ00</yt:videoId>
    <title>Claude Code + MCP with TypeScript AI SDK</title>
    <link rel="alternate" href="https://www.youtube.com/watch?v=abc123XYZ00"/>
    <published>2026-11-20T15:00:00+00:00</published>
    <media:group>
      <media:description>Deep dive on Claude Code, MCP servers, and Vercel AI SDK evals.</media:description>
      <media:thumbnail url="https://i.ytimg.com/vi/abc123XYZ00/hqdefault.jpg"/>
    </media:group>
  </entry>
</feed>
"""


def test_channels_yaml_loads_all_14_channels() -> None:
    channels = load_channels(Path("channels.yaml"))
    assert len(channels) == 14
    ids = {c.channel_id for c in channels}
    assert len(ids) == 14
    assert all(cid.startswith("UC") for cid in ids)


def test_parse_youtube_atom_feed() -> None:
    videos = parse_youtube_atom_feed(
        SAMPLE_ATOM,
        channel_id="UCswG6FSbgZjbWtdf_hMLaow",
        fallback_channel_name="Matt Pocock",
    )
    assert len(videos) == 1
    v = videos[0]
    assert v.video_id == "abc123XYZ00"
    assert v.channel_name == "Matt Pocock"
    assert "Claude Code" in v.title
    assert v.url == "https://www.youtube.com/watch?v=abc123XYZ00"


def test_entity_extraction_and_heuristic_analysis() -> None:
    entities = extract_known_entities(
        "Comparing Claude Code and Cursor with MCP servers in TypeScript using AI SDK."
    )
    names = {e["name"] for e in entities}
    assert {"Claude Code", "Cursor", "MCP", "TypeScript", "AI SDK"}.issubset(names)

    result = analyze_heuristic(
        title="Claude Code + MCP",
        channel_name="Matt Pocock",
        description="Building reliable coding agents with MCP and evals.",
        transcript_text="Today we test Claude Code with MCP servers and evals.",
    )
    assert result.status == "heuristic"
    assert result.summary
    assert any(e["name"] == "MCP" for e in result.entities)


def test_db_fts_trends_and_web_routes(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        channels_file=Path("channels.yaml"),
        enable_scheduler=False,
        llm_api_key="",
    )
    db = Database(settings.db_path)
    channels = load_channels(settings.channels_file)
    db.upsert_channels(channels)

    # Seed two channels talking about Claude Code & MCP
    db.upsert_discovered_video(
        video_id="vid_matt_1",
        channel_id="UCswG6FSbgZjbWtdf_hMLaow",
        channel_name="Matt Pocock",
        title="Mastering Claude Code and MCP",
        url="https://www.youtube.com/watch?v=vid_matt_1",
        published_at="2026-11-23T12:00:00+00:00",
        description="TypeScript agent workflows with MCP.",
        thumbnail_url="https://i.ytimg.com/vi/vid_matt_1/hqdefault.jpg",
    )
    db.save_transcript(
        "vid_matt_1",
        status="ok",
        source="test",
        language="en",
        text="We wire up Claude Code with Model Context Protocol and evals.",
        segments=[{"start": 12.0, "duration": 4.0, "text": "We wire up Claude Code."}],
    )
    db.save_analysis(
        "vid_matt_1",
        summary="How to structure TypeScript agent workflows with Claude Code and MCP.",
        takeaways=["Keep MCP tool schemas small", "Run automated evals"],
        key_moments=[{"timestamp_sec": 12, "label": "MCP setup"}],
        entities=[
            {"name": "Claude Code", "type": "tool"},
            {"name": "MCP", "type": "protocol"},
        ],
        sentiment="tutorial",
        status="llm",
        model="test-model",
    )

    db.upsert_discovered_video(
        video_id="vid_theo_1",
        channel_id="UCbRP3c757lWg9M-U7TyEkXA",
        channel_name="Theo - t3.gg",
        title="Claude Code Changed Everything",
        url="https://www.youtube.com/watch?v=vid_theo_1",
        published_at="2026-11-24T10:00:00+00:00",
        description="Why developers are switching to terminal coding agents.",
        thumbnail_url="https://i.ytimg.com/vi/vid_theo_1/hqdefault.jpg",
    )
    db.save_analysis(
        "vid_theo_1",
        summary="Theo breaks down why Claude Code is winning over IDE workflows.",
        takeaways=["Terminal-first agents compose cleanly with git"],
        key_moments=[],
        entities=[{"name": "Claude Code", "type": "tool"}],
        sentiment="bullish",
        status="llm",
        model="test-model",
    )

    # FTS search on transcript content
    found = db.list_videos(q="Protocol")
    assert [v["video_id"] for v in found] == ["vid_matt_1"]

    # Trend radar marks Claude Code as hot (2 channels)
    trends = db.get_trends(days=30)
    claude_trend = next(t for t in trends if t["name"] == "Claude Code")
    assert claude_trend["hot"] is True
    assert claude_trend["channel_count"] == 2
    assert claude_trend["video_count"] == 2

    # FastAPI endpoints & templates render cleanly
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/").status_code == 200
        assert "Mastering Claude Code" in client.get("/").text
        assert client.get("/trends").status_code == 200
        assert client.get("/channels").status_code == 200
        assert client.get("/video/vid_matt_1").status_code == 200
        assert client.get("/api/trends").json()["trends"][0]["name"] == "Claude Code"
        feed = client.get("/feed.xml")
        assert feed.status_code == 200
        assert "<feed" in feed.text
