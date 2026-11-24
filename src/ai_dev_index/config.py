from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChannelConfig(BaseModel):
    name: str
    handle: str = ""
    channel_id: str
    category: str = "ai-engineering"
    priority: Literal["high", "medium", "low"] = "medium"

    @property
    def youtube_url(self) -> str:
        if self.handle:
            return f"https://www.youtube.com/{self.handle}"
        return f"https://www.youtube.com/channel/{self.channel_id}"

    @property
    def rss_url(self) -> str:
        return f"https://www.youtube.com/feeds/videos.xml?channel_id={self.channel_id}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("AI_DEV_INDEX_HOST", "HOST"),
    )
    port: int = Field(
        default=8080,
        validation_alias=AliasChoices("AI_DEV_INDEX_PORT", "PORT"),
    )
    data_dir: Path = Field(
        default=Path("./data"),
        validation_alias=AliasChoices("AI_DEV_INDEX_DATA_DIR", "DATA_DIR"),
    )
    channels_file: Path = Field(
        default=Path("./channels.yaml"),
        validation_alias=AliasChoices("AI_DEV_INDEX_CHANNELS_FILE", "CHANNELS_FILE"),
    )

    sync_interval_minutes: int = Field(
        default=180,
        validation_alias=AliasChoices("AI_DEV_INDEX_SYNC_INTERVAL_MINUTES"),
    )
    videos_per_channel: int = Field(
        default=5,
        validation_alias=AliasChoices("AI_DEV_INDEX_VIDEOS_PER_CHANNEL"),
    )
    request_delay_seconds: float = Field(
        default=1.5,
        validation_alias=AliasChoices("AI_DEV_INDEX_REQUEST_DELAY_SECONDS"),
    )
    enable_scheduler: bool = Field(
        default=True,
        validation_alias=AliasChoices("AI_DEV_INDEX_ENABLE_SCHEDULER"),
    )
    store_transcripts: bool = Field(
        default=True,
        validation_alias=AliasChoices("AI_DEV_INDEX_STORE_TRANSCRIPTS"),
    )

    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_MODEL"),
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS"),
    )

    @property
    def db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "ai-dev-index.db"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key.strip())


def load_channels(path: Path) -> list[ChannelConfig]:
    if not path.exists():
        raise FileNotFoundError(f"Channels file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("channels", [])
    return [ChannelConfig.model_validate(item) for item in items]
