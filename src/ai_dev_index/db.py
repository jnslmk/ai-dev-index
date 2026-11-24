from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_dev_index.config import ChannelConfig


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS channels (
    channel_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    handle TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'ai-engineering',
    priority TEXT NOT NULL DEFAULT 'medium',
    last_synced_at TEXT
);

CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    channel_name TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    thumbnail_url TEXT NOT NULL DEFAULT '',
    transcript_status TEXT NOT NULL DEFAULT 'pending',
    transcript_source TEXT NOT NULL DEFAULT '',
    transcript_language TEXT NOT NULL DEFAULT '',
    transcript_text TEXT NOT NULL DEFAULT '',
    transcript_segments_json TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '',
    takeaways_json TEXT NOT NULL DEFAULT '[]',
    key_moments_json TEXT NOT NULL DEFAULT '[]',
    sentiment TEXT NOT NULL DEFAULT '',
    analysis_status TEXT NOT NULL DEFAULT 'pending',
    analysis_model TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    discovered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_analysis_status ON videos(analysis_status);

CREATE TABLE IF NOT EXISTS video_entities (
    video_id TEXT NOT NULL,
    entity TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'topic',
    PRIMARY KEY (video_id, entity),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_video_entities_entity ON video_entities(entity);

CREATE VIRTUAL TABLE IF NOT EXISTS videos_fts USING fts5(
    video_id UNINDEXED,
    title,
    channel_name,
    summary,
    takeaways_text,
    entities_text,
    transcript_text,
    tokenize = 'unicode61'
);

CREATE TABLE IF NOT EXISTS sync_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    channels_checked INTEGER NOT NULL DEFAULT 0,
    videos_discovered INTEGER NOT NULL DEFAULT 0,
    transcripts_fetched INTEGER NOT NULL DEFAULT 0,
    videos_analyzed INTEGER NOT NULL DEFAULT 0,
    errors_json TEXT NOT NULL DEFAULT '[]'
);
"""


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def upsert_channels(self, channels: list[ChannelConfig]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO channels (channel_id, name, handle, category, priority)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    name = excluded.name,
                    handle = excluded.handle,
                    category = excluded.category,
                    priority = excluded.priority
                """,
                [
                    (c.channel_id, c.name, c.handle, c.category, c.priority)
                    for c in channels
                ],
            )

    def mark_channel_synced(self, channel_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE channels SET last_synced_at = ? WHERE channel_id = ?",
                (utc_now_iso(), channel_id),
            )

    def upsert_discovered_video(
        self,
        *,
        video_id: str,
        channel_id: str,
        channel_name: str,
        title: str,
        url: str,
        published_at: str,
        description: str,
        thumbnail_url: str,
    ) -> bool:
        """Insert a video if unseen; update metadata if seen. Returns True if newly inserted."""
        now = utc_now_iso()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE videos
                    SET title = ?, description = ?, thumbnail_url = ?, updated_at = ?
                    WHERE video_id = ?
                    """,
                    (title, description, thumbnail_url, now, video_id),
                )
                return False

            conn.execute(
                """
                INSERT INTO videos (
                    video_id, channel_id, channel_name, title, url,
                    published_at, description, thumbnail_url,
                    discovered_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    channel_id,
                    channel_name,
                    title,
                    url,
                    published_at,
                    description,
                    thumbnail_url,
                    now,
                    now,
                ),
            )
            self._refresh_fts_row(conn, video_id)
            return True

    def save_transcript(
        self,
        video_id: str,
        *,
        status: str,
        source: str = "",
        language: str = "",
        text: str = "",
        segments: list[dict[str, Any]] | None = None,
        store_text: bool = True,
        error_message: str = "",
    ) -> None:
        now = utc_now_iso()
        stored_text = text if store_text else ""
        stored_segments = json.dumps(segments or []) if store_text else "[]"
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE videos
                SET transcript_status = ?,
                    transcript_source = ?,
                    transcript_language = ?,
                    transcript_text = ?,
                    transcript_segments_json = ?,
                    error_message = ?,
                    updated_at = ?
                WHERE video_id = ?
                """,
                (
                    status,
                    source,
                    language,
                    stored_text,
                    stored_segments,
                    error_message,
                    now,
                    video_id,
                ),
            )
            self._refresh_fts_row(conn, video_id)

    def save_analysis(
        self,
        video_id: str,
        *,
        summary: str,
        takeaways: list[str],
        key_moments: list[dict[str, Any]],
        entities: list[dict[str, str]],
        sentiment: str,
        status: str,
        model: str,
        error_message: str = "",
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE videos
                SET summary = ?,
                    takeaways_json = ?,
                    key_moments_json = ?,
                    sentiment = ?,
                    analysis_status = ?,
                    analysis_model = ?,
                    error_message = ?,
                    updated_at = ?
                WHERE video_id = ?
                """,
                (
                    summary,
                    json.dumps(takeaways),
                    json.dumps(key_moments),
                    sentiment,
                    status,
                    model,
                    error_message,
                    now,
                    video_id,
                ),
            )
            conn.execute("DELETE FROM video_entities WHERE video_id = ?", (video_id,))
            seen: set[str] = set()
            entity_rows: list[tuple[str, str, str]] = []
            for item in entities:
                name = item.get("name", "").strip()
                etype = item.get("type", "topic").strip() or "topic"
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    entity_rows.append((video_id, name, etype))
            if entity_rows:
                conn.executemany(
                    "INSERT OR IGNORE INTO video_entities (video_id, entity, entity_type) VALUES (?, ?, ?)",
                    entity_rows,
                )
            self._refresh_fts_row(conn, video_id)

    def _refresh_fts_row(self, conn: sqlite3.Connection, video_id: str) -> None:
        row = conn.execute(
            """
            SELECT video_id, title, channel_name, summary, takeaways_json, transcript_text
            FROM videos WHERE video_id = ?
            """,
            (video_id,),
        ).fetchone()
        if not row:
            return
        entity_rows = conn.execute(
            "SELECT entity FROM video_entities WHERE video_id = ?", (video_id,)
        ).fetchall()
        takeaways = json.loads(row["takeaways_json"] or "[]")
        takeaways_text = " ".join(str(t) for t in takeaways)
        entities_text = " ".join(r["entity"] for r in entity_rows)
        conn.execute("DELETE FROM videos_fts WHERE video_id = ?", (video_id,))
        conn.execute(
            """
            INSERT INTO videos_fts (
                video_id, title, channel_name, summary,
                takeaways_text, entities_text, transcript_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["video_id"],
                row["title"],
                row["channel_name"],
                row["summary"],
                takeaways_text,
                entities_text,
                row["transcript_text"],
            ),
        )

    def get_video(self, video_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT v.*, c.handle AS channel_handle, c.category AS channel_category
                FROM videos v
                LEFT JOIN channels c ON c.channel_id = v.channel_id
                WHERE v.video_id = ?
                """,
                (video_id,),
            ).fetchone()
            if not row:
                return None
            return self._hydrate_video(conn, row)

    def list_videos(
        self,
        *,
        q: str = "",
        channel_id: str = "",
        category: str = "",
        entity: str = "",
        days: int | None = None,
        limit: int = 40,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        joins: list[str] = ["LEFT JOIN channels c ON c.channel_id = v.channel_id"]

        if q.strip():
            tokens = [tok.replace('"', "") for tok in q.strip().split() if tok.strip()]
            if tokens:
                fts_query = " AND ".join(f'"{tok}"*' for tok in tokens)
                joins.append("JOIN videos_fts fts ON fts.video_id = v.video_id")
                clauses.append("videos_fts MATCH ?")
                params.append(fts_query)

        if channel_id:
            clauses.append("v.channel_id = ?")
            params.append(channel_id)

        if category:
            clauses.append("c.category = ?")
            params.append(category)

        if entity:
            joins.append("JOIN video_entities ve ON ve.video_id = v.video_id")
            clauses.append("LOWER(ve.entity) = LOWER(?)")
            params.append(entity)

        if days is not None and days > 0:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            clauses.append("v.published_at >= ?")
            params.append(cutoff)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT DISTINCT v.*, c.handle AS channel_handle, c.category AS channel_category
            FROM videos v
            {' '.join(joins)}
            {where_sql}
            ORDER BY v.published_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._hydrate_video(conn, row, include_transcript=False) for row in rows]

    def list_videos_needing_processing(
        self,
        *,
        include_heuristic: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        statuses = ["pending", "error"]
        if include_heuristic:
            statuses.append("heuristic")
        placeholders = ",".join("?" for _ in statuses)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT v.*, c.handle AS channel_handle, c.category AS channel_category
                FROM videos v
                LEFT JOIN channels c ON c.channel_id = v.channel_id
                WHERE v.analysis_status IN ({placeholders})
                ORDER BY v.published_at DESC
                LIMIT ?
                """,
                [*statuses, limit],
            ).fetchall()
            return [self._hydrate_video(conn, row, include_transcript=True) for row in rows]

    def _hydrate_video(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        include_transcript: bool = True,
    ) -> dict[str, Any]:
        item = dict(row)
        item["takeaways"] = json.loads(item.pop("takeaways_json", "[]") or "[]")
        item["key_moments"] = json.loads(item.pop("key_moments_json", "[]") or "[]")
        segments_raw = item.pop("transcript_segments_json", "[]") or "[]"
        if include_transcript:
            item["transcript_segments"] = json.loads(segments_raw)
        else:
            item["transcript_segments"] = []
            item["transcript_word_count"] = len((item.get("transcript_text") or "").split())
            item["transcript_text"] = ""

        ent_rows = conn.execute(
            "SELECT entity AS name, entity_type AS type FROM video_entities WHERE video_id = ? ORDER BY entity",
            (item["video_id"],),
        ).fetchall()
        item["entities"] = [dict(r) for r in ent_rows]
        return item

    def get_trends(self, *, days: int = 14, limit: int = 24) -> list[dict[str, Any]]:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    ve.entity AS name,
                    MAX(ve.entity_type) AS type,
                    COUNT(DISTINCT ve.video_id) AS video_count,
                    COUNT(DISTINCT v.channel_id) AS channel_count,
                    MAX(v.published_at) AS latest_published_at,
                    GROUP_CONCAT(DISTINCT v.channel_name) AS channels_csv
                FROM video_entities ve
                JOIN videos v ON v.video_id = ve.video_id
                WHERE v.published_at >= ?
                GROUP BY LOWER(ve.entity)
                ORDER BY channel_count DESC, video_count DESC, latest_published_at DESC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()

            results = []
            for r in rows:
                channels_list = sorted(
                    c.strip() for c in (r["channels_csv"] or "").split(",") if c.strip()
                )
                results.append(
                    {
                        "name": r["name"],
                        "type": r["type"],
                        "video_count": r["video_count"],
                        "channel_count": r["channel_count"],
                        "hot": r["channel_count"] >= 2,
                        "latest_published_at": r["latest_published_at"],
                        "channels": channels_list,
                    }
                )
            return results

    def list_channels_with_counts(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.*,
                    COUNT(v.video_id) AS video_count,
                    MAX(v.published_at) AS latest_video_at
                FROM channels c
                LEFT JOIN videos v ON v.channel_id = c.channel_id
                GROUP BY c.channel_id
                ORDER BY
                    CASE c.priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                    c.name ASC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            totals = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM channels) AS channels,
                    COUNT(*) AS videos,
                    SUM(CASE WHEN transcript_status = 'ok' THEN 1 ELSE 0 END) AS transcripts_ok,
                    SUM(CASE WHEN analysis_status = 'llm' THEN 1 ELSE 0 END) AS llm_summaries,
                    SUM(CASE WHEN analysis_status = 'heuristic' THEN 1 ELSE 0 END) AS heuristic_summaries
                FROM videos
                """
            ).fetchone()
            last_run = conn.execute(
                "SELECT * FROM sync_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return {
                "channels": totals["channels"] or 0,
                "videos": totals["videos"] or 0,
                "transcripts_ok": totals["transcripts_ok"] or 0,
                "llm_summaries": totals["llm_summaries"] or 0,
                "heuristic_summaries": totals["heuristic_summaries"] or 0,
                "last_run": dict(last_run) if last_run else None,
            }

    def start_sync_run(self) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO sync_runs (started_at, status) VALUES (?, 'running')",
                (utc_now_iso(),),
            )
            return int(cur.lastrowid)

    def finish_sync_run(
        self,
        run_id: int,
        *,
        status: str,
        channels_checked: int,
        videos_discovered: int,
        transcripts_fetched: int,
        videos_analyzed: int,
        errors: list[str],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE sync_runs
                SET finished_at = ?,
                    status = ?,
                    channels_checked = ?,
                    videos_discovered = ?,
                    transcripts_fetched = ?,
                    videos_analyzed = ?,
                    errors_json = ?
                WHERE id = ?
                """,
                (
                    utc_now_iso(),
                    status,
                    channels_checked,
                    videos_discovered,
                    transcripts_fetched,
                    videos_analyzed,
                    json.dumps(errors),
                    run_id,
                ),
            )
