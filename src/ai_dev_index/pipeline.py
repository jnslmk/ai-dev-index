from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import httpx

from ai_dev_index.analyzer import analyze_video
from ai_dev_index.config import ChannelConfig, Settings, load_channels
from ai_dev_index.db import Database
from ai_dev_index.ingest import fetch_channel_feed, fetch_transcript

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SyncSummary:
    run_id: int
    channels_checked: int = 0
    videos_discovered: int = 0
    transcripts_fetched: int = 0
    videos_analyzed: int = 0
    errors: list[str] = field(default_factory=list)


class SyncPipeline:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._lock.locked()

    def load_configured_channels(self) -> list[ChannelConfig]:
        channels = load_channels(self.settings.channels_file)
        self.db.upsert_channels(channels)
        return channels

    def sync(
        self,
        *,
        limit_per_channel: int | None = None,
        channel_filter: str | None = None,
        skip_transcripts: bool = False,
        reanalyze_heuristic: bool = False,
    ) -> SyncSummary:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A sync run is already in progress")

        run_id = self.db.start_sync_run()
        summary = SyncSummary(run_id=run_id)
        per_channel = limit_per_channel or self.settings.videos_per_channel

        try:
            channels = self.load_configured_channels()
            if channel_filter:
                needle = channel_filter.lower()
                channels = [
                    c
                    for c in channels
                    if needle in c.name.lower()
                    or needle in c.handle.lower()
                    or needle == c.channel_id.lower()
                ]

            with httpx.Client(follow_redirects=True) as client:
                for channel in channels:
                    summary.channels_checked += 1
                    try:
                        videos = fetch_channel_feed(
                            channel.channel_id,
                            channel_name=channel.name,
                            client=client,
                            limit=per_channel,
                        )
                        self.db.mark_channel_synced(channel.channel_id)
                    except Exception as exc:  # noqa: BLE001
                        msg = f"RSS failed for {channel.name}: {exc}"
                        logger.warning(msg)
                        summary.errors.append(msg)
                        continue

                    for video in videos:
                        is_new = self.db.upsert_discovered_video(
                            video_id=video.video_id,
                            channel_id=video.channel_id,
                            channel_name=video.channel_name,
                            title=video.title,
                            url=video.url,
                            published_at=video.published_at,
                            description=video.description,
                            thumbnail_url=video.thumbnail_url,
                        )
                        if is_new:
                            summary.videos_discovered += 1

                        existing = self.db.get_video(video.video_id)
                        if not existing:
                            continue

                        transcript_text = existing["transcript_text"]
                        transcript_segments = existing["transcript_segments"]

                        if not skip_transcripts and existing["transcript_status"] in {
                            "pending",
                            "error",
                        }:
                            t_res = fetch_transcript(video.video_id)
                            self.db.save_transcript(
                                video.video_id,
                                status=t_res.status,
                                source=t_res.source,
                                language=t_res.language,
                                text=t_res.text,
                                segments=t_res.segments,
                                store_text=self.settings.store_transcripts,
                                error_message=t_res.error,
                            )
                            if t_res.status == "ok":
                                summary.transcripts_fetched += 1
                                transcript_text = t_res.text
                                transcript_segments = t_res.segments
                            if self.settings.request_delay_seconds > 0:
                                time.sleep(self.settings.request_delay_seconds)

                        needs_analysis = existing["analysis_status"] in {"pending", "error"} or (
                            reanalyze_heuristic
                            and self.settings.llm_enabled
                            and existing["analysis_status"] == "heuristic"
                        )
                        if needs_analysis:
                            a_res = analyze_video(
                                title=existing["title"],
                                channel_name=existing["channel_name"],
                                description=existing["description"],
                                transcript_text=transcript_text,
                                transcript_segments=transcript_segments,
                                settings=self.settings,
                            )
                            self.db.save_analysis(
                                video.video_id,
                                summary=a_res.summary,
                                takeaways=a_res.takeaways,
                                key_moments=a_res.key_moments,
                                entities=a_res.entities,
                                sentiment=a_res.sentiment,
                                status=a_res.status,
                                model=a_res.model,
                                error_message=a_res.error,
                            )
                            summary.videos_analyzed += 1

            self.db.finish_sync_run(
                run_id,
                status="ok" if not summary.errors else "partial",
                channels_checked=summary.channels_checked,
                videos_discovered=summary.videos_discovered,
                transcripts_fetched=summary.transcripts_fetched,
                videos_analyzed=summary.videos_analyzed,
                errors=summary.errors,
            )
            return summary
        except Exception as exc:
            summary.errors.append(str(exc))
            self.db.finish_sync_run(
                run_id,
                status="error",
                channels_checked=summary.channels_checked,
                videos_discovered=summary.videos_discovered,
                transcripts_fetched=summary.transcripts_fetched,
                videos_analyzed=summary.videos_analyzed,
                errors=summary.errors,
            )
            raise
        finally:
            self._lock.release()
