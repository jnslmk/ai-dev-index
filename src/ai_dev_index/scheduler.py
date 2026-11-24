from __future__ import annotations

import asyncio
import logging

from ai_dev_index.config import Settings
from ai_dev_index.pipeline import SyncPipeline

logger = logging.getLogger(__name__)


async def run_periodic_sync(pipeline: SyncPipeline, settings: Settings) -> None:
    interval_sec = max(60, settings.sync_interval_minutes * 60)
    logger.info(
        "Background sync scheduler enabled (every %d minutes)",
        settings.sync_interval_minutes,
    )
    # Wait briefly on startup so the web UI becomes responsive immediately
    await asyncio.sleep(10)
    while True:
        try:
            logger.info("Starting scheduled channel sync...")
            summary = await asyncio.to_thread(pipeline.sync)
            logger.info(
                "Scheduled sync complete: channels=%d discovered=%d transcripts=%d analyzed=%d errors=%d",
                summary.channels_checked,
                summary.videos_discovered,
                summary.transcripts_fetched,
                summary.videos_analyzed,
                len(summary.errors),
            )
        except RuntimeError as exc:
            logger.info("Skipping scheduled sync: %s", exc)
        except Exception:
            logger.exception("Scheduled sync failed")

        await asyncio.sleep(interval_sec)
