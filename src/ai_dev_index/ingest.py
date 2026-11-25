from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}


class TranscriptRateLimited(Exception):
    """YouTube is throttling/blocking caption requests for this IP."""


@dataclass(slots=True)
class DiscoveredVideo:
    video_id: str
    channel_id: str
    channel_name: str
    title: str
    url: str
    published_at: str
    description: str
    thumbnail_url: str


@dataclass(slots=True)
class TranscriptResult:
    status: str  # ok | unavailable | error
    source: str = ""
    language: str = ""
    text: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


def parse_youtube_atom_feed(
    xml_bytes: bytes,
    *,
    channel_id: str,
    fallback_channel_name: str = "",
) -> list[DiscoveredVideo]:
    root = ET.fromstring(xml_bytes)
    feed_title = root.findtext("atom:title", default="", namespaces=ATOM_NS).strip()
    channel_name = fallback_channel_name or feed_title or channel_id

    videos: list[DiscoveredVideo] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        video_id = entry.findtext("yt:videoId", default="", namespaces=ATOM_NS).strip()
        if not video_id:
            continue
        title = entry.findtext("atom:title", default="", namespaces=ATOM_NS).strip()
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip()

        link_el = entry.find("atom:link[@rel='alternate']", ATOM_NS)
        url = (
            link_el.attrib.get("href", "")
            if link_el is not None
            else f"https://www.youtube.com/watch?v={video_id}"
        )

        media_group = entry.find("media:group", ATOM_NS)
        description = ""
        thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        if media_group is not None:
            description = (
                media_group.findtext("media:description", default="", namespaces=ATOM_NS) or ""
            ).strip()
            thumb_el = media_group.find("media:thumbnail", ATOM_NS)
            if thumb_el is not None and thumb_el.attrib.get("url"):
                thumbnail_url = thumb_el.attrib["url"]

        videos.append(
            DiscoveredVideo(
                video_id=video_id,
                channel_id=channel_id,
                channel_name=channel_name,
                title=title,
                url=url,
                published_at=published,
                description=description,
                thumbnail_url=thumbnail_url,
            )
        )
    return videos


def fetch_channel_feed(
    channel_id: str,
    *,
    channel_name: str = "",
    client: httpx.Client | None = None,
    limit: int = 10,
) -> list[DiscoveredVideo]:
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    headers = {"User-Agent": "ai-dev-index/0.1 (+self-hosted-rss)"}

    if client is not None:
        resp = client.get(url, headers=headers, timeout=20.0)
        resp.raise_for_status()
        xml_bytes = resp.content
    else:
        with httpx.Client(follow_redirects=True) as local_client:
            resp = local_client.get(url, headers=headers, timeout=20.0)
            resp.raise_for_status()
            xml_bytes = resp.content

    videos = parse_youtube_atom_feed(
        xml_bytes,
        channel_id=channel_id,
        fallback_channel_name=channel_name,
    )
    return videos[:limit]


def _segments_to_text(segments: list[dict[str, Any]]) -> str:
    parts = []
    for seg in segments:
        txt = " ".join(str(seg.get("text", "")).split())
        if txt:
            parts.append(txt)
    return " ".join(parts)


def _fetch_via_youtube_transcript_api(
    video_id: str,
    languages: tuple[str, ...],
) -> TranscriptResult | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None

    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):
            fetched = api.fetch(video_id, languages=list(languages))
            raw_segments = [
                {
                    "start": round(float(getattr(s, "start", 0.0)), 2),
                    "duration": round(float(getattr(s, "duration", 0.0)), 2),
                    "text": str(getattr(s, "text", "")).strip(),
                }
                for s in fetched
            ]
            lang = getattr(fetched, "language_code", languages[0])
        else:
            # Fallback for pre-1.0 youtube-transcript-api
            fetched_old = YouTubeTranscriptApi.get_transcript(  # type: ignore[attr-defined]
                video_id, languages=list(languages)
            )
            raw_segments = [
                {
                    "start": round(float(s.get("start", 0.0)), 2),
                    "duration": round(float(s.get("duration", 0.0)), 2),
                    "text": str(s.get("text", "")).strip(),
                }
                for s in fetched_old
            ]
            lang = languages[0]

        text = _segments_to_text(raw_segments)
        if not text:
            return None
        return TranscriptResult(
            status="ok",
            source="youtube-transcript-api",
            language=lang,
            text=text,
            segments=raw_segments,
        )
    except Exception as exc:
        if type(exc).__name__ in {"RequestBlocked", "IpBlocked"} or "429" in str(exc):
            raise TranscriptRateLimited(str(exc)) from exc
        logger.debug("youtube-transcript-api failed for %s: %s", video_id, exc)
        return None


def _parse_json3_subtitles(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        if not text:
            continue
        start = round(float(event.get("tStartMs", 0)) / 1000.0, 2)
        duration = round(float(event.get("dDurationMs", 0)) / 1000.0, 2)
        segments.append({"start": start, "duration": duration, "text": text})
    return segments


def _fetch_via_yt_dlp(
    video_id: str,
    languages: tuple[str, ...],
) -> TranscriptResult | None:
    url = f"https://www.youtube.com/watch?v={video_id}"
    lang_arg = ",".join(languages)
    with tempfile.TemporaryDirectory(prefix="ai-dev-index-") as tmp:
        out_template = str(Path(tmp) / "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--no-warnings",
            "--quiet",
            "--write-sub",
            "--write-auto-sub",
            "--sub-langs",
            lang_arg,
            "--sub-format",
            "json3",
            "-o",
            out_template,
            url,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("yt-dlp failed for %s: %s", video_id, exc)
            return None

        sub_files = sorted(Path(tmp).glob(f"{video_id}*.json3"))
        if not sub_files:
            stderr = proc.stderr.strip()
            if (
                "429" in stderr
                or "Too Many Requests" in stderr
                or "confirm you're not a bot" in stderr
            ):
                raise TranscriptRateLimited(stderr[:300])
            logger.debug(
                "yt-dlp found no subtitles for %s (rc=%s, stderr=%s)",
                video_id,
                proc.returncode,
                stderr[:200],
            )
            return None

        sub_file = sub_files[0]
        segments = _parse_json3_subtitles(sub_file)
        text = _segments_to_text(segments)
        if not text:
            return None
        parts = sub_file.name.split(".")
        lang = parts[1] if len(parts) >= 3 else languages[0]
        return TranscriptResult(
            status="ok",
            source="yt-dlp",
            language=lang,
            text=text,
            segments=segments,
        )


def fetch_transcript(
    video_id: str,
    *,
    languages: tuple[str, ...] = ("en", "en-US", "en-GB"),
) -> TranscriptResult:
    """Fetch transcript using youtube-transcript-api, falling back to yt-dlp.

    Raises TranscriptRateLimited when YouTube throttles the request, so the
    caller can stop hitting the API for the rest of the run instead of walking
    every remaining video into the same 429.
    """
    res = _fetch_via_youtube_transcript_api(video_id, languages)
    if res is not None:
        return res

    res = _fetch_via_yt_dlp(video_id, languages)
    if res is not None:
        return res

    return TranscriptResult(
        status="unavailable",
        error="No English captions available via youtube-transcript-api or yt-dlp",
    )
