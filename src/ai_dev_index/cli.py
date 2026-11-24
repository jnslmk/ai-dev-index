from __future__ import annotations

import argparse
import json
import logging
import sys

import uvicorn

from ai_dev_index.config import Settings
from ai_dev_index.db import Database
from ai_dev_index.pipeline import SyncPipeline


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-dev-index",
        description="Self-hosted index, summaries, and trend radar for AI engineering YouTube channels",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve
    serve_p = subparsers.add_parser("serve", help="Run the internal web dashboard and scheduler")
    serve_p.add_argument("--host", default=None)
    serve_p.add_argument("--port", type=int, default=None)

    # sync
    sync_p = subparsers.add_parser("sync", help="Fetch latest videos, transcripts, and summaries")
    sync_p.add_argument("--limit-per-channel", type=int, default=None)
    sync_p.add_argument("--channel", default=None, help="Filter by channel name, handle, or ID")
    sync_p.add_argument("--skip-transcripts", action="store_true")
    sync_p.add_argument(
        "--reanalyze-heuristic",
        action="store_true",
        help="Upgrade heuristic entries using the configured LLM",
    )

    # stats
    subparsers.add_parser("stats", help="Show index statistics as JSON")

    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    settings = Settings()

    if args.command == "serve":
        uvicorn.run(
            "ai_dev_index.web:create_app",
            factory=True,
            host=args.host or settings.host,
            port=args.port or settings.port,
            log_level="debug" if args.verbose else "info",
        )
        return 0

    db = Database(settings.db_path)
    pipeline = SyncPipeline(db, settings)

    if args.command == "sync":
        summary = pipeline.sync(
            limit_per_channel=args.limit_per_channel,
            channel_filter=args.channel,
            skip_transcripts=args.skip_transcripts,
            reanalyze_heuristic=args.reanalyze_heuristic,
        )
        print(
            json.dumps(
                {
                    "run_id": summary.run_id,
                    "channels_checked": summary.channels_checked,
                    "videos_discovered": summary.videos_discovered,
                    "transcripts_fetched": summary.transcripts_fetched,
                    "videos_analyzed": summary.videos_analyzed,
                    "errors": summary.errors,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "stats":
        pipeline.load_configured_channels()
        print(json.dumps(db.get_stats(), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
