# ai-dev-index

A self-hosted index, structured summary dashboard, and **Trend Radar** for high-signal AI software engineering YouTube channels.

Designed to run on a **NAS or home server** (avoiding YouTube's datacenter/CI IP blocks), storing everything in a single local SQLite database with FTS5 full-text search and serving a fast, zero-CDN internal web UI + Atom RSS feed.

## Features

- **RSS-first discovery:** Polls YouTube's official Atom feeds (`youtube.com/feeds/videos.xml`) — no YouTube API key or quota required.
- **Resilient caption fetching:** Uses `youtube-transcript-api` with automatic `yt-dlp` subtitle fallback and polite rate-limiting.
- **Provider-agnostic LLM analysis:** Works with any OpenAI-compatible API (OpenAI, OpenRouter, local Ollama on your NAS, LiteLLM, Gemini OpenAI-compat). Generates fresh, original summaries, engineering takeaways, key timestamped moments, and normalized entities.
- **Works out of the box without an LLM key:** Falls back to deterministic AI-dev entity extraction + local FTS5 search so you can index immediately and upgrade summaries later with `--reanalyze-heuristic`.
- **Trend Radar & Cross-Creator Comparison:** Tracks tool/model/framework mentions over 7/14/30/90-day windows and flags **hot topics** covered by 2+ channels.
- **Internal Web UI + JSON API + Atom RSS (`/feed.xml`):** Browse by channel, category, or topic, search locally across summaries and transcripts, or subscribe in your RSS reader.

## Pre-configured Channels (`channels.yaml`)

- **Core AI Engineering:** Matt Pocock, Theo - t3.gg, Better Stack
- **Coding Agents & Production Talks:** AI Engineer, IndyDevDan, Dave Ebbelaar, Cole Medin, Latent Space, Fireship
- **Analysis & Foundations:** AI Explained, Andrej Karpathy
- **Official Lab & Framework Channels:** Anthropic, OpenAI, LangChain

Edit [`channels.yaml`](./channels.yaml) to add or remove channels at any time.

## Quick Start (Local)

```bash
cp .env.example .env
# Optional: set LLM_API_KEY (or point LLM_BASE_URL at local Ollama)

# Sync the latest 3 videos per channel
uv run ai-dev-index sync --limit-per-channel 3

# Start the internal web dashboard + background scheduler
uv run ai-dev-index serve
```

Open `http://127.0.0.1:8080`.

## CI/CD & NAS Deployment (Forgejo Registry)

The canonical repository is on Forgejo at `forgejo.lemke.dev/jonas/ai-dev-index`.

Images are automatically built on push to `main` by [`.forgejo/workflows/ci.yml`](./.forgejo/workflows/ci.yml) and published to the Forgejo container registry:

- `forgejo.lemke.dev/jonas/ai-dev-index:0.1.<run_number>`
- `forgejo.lemke.dev/jonas/ai-dev-index:sha-<short_sha>`
- `forgejo.lemke.dev/jonas/ai-dev-index:latest`

On the NAS, the service is managed declaratively via `jnslmk/nas-ansible` at `ai-dev-index.internal.lemke.dev`. The NAS pin is bumped by the same release dispatch that triggers the Forgejo CI build, so a merge to `main` rolls out through the usual `nas-ansible` renovate → auto-deploy path.

For standalone Docker Compose usage:

```bash
cp .env.example .env
# Edit .env with your preferred LLM settings & sync interval

docker compose up -d --build
```

- Persistent SQLite database is stored in `./data/ai-dev-index.db` (ignored by git).
- `docker-compose.yml` mounts `./channels.yaml` over the baked-in copy, so editing the host file and restarting picks up new channels without a rebuild. The NAS deployment bakes the channel list into the image; change `channels.yaml` in the repo and redeploy there.
- Works great behind Tailscale, Caddy, Nginx Proxy Manager, or Traefik on your LAN.

## CLI Reference

```bash
# Run a one-off sync
uv run ai-dev-index sync --limit-per-channel 5

# Sync a single channel
uv run ai-dev-index sync --channel "Matt Pocock"

# Upgrade heuristic entries once you add an LLM_API_KEY
uv run ai-dev-index sync --reanalyze-heuristic

# Inspect index stats
uv run ai-dev-index stats

# Run web UI + periodic background sync
uv run ai-dev-index serve --host 0.0.0.0 --port 8080
```

## Privacy & Copyright Note

- This repository contains **code and channel config only** — never raw transcripts or local databases (`data/` is gitignored).
- For **internal self-hosted use**, `AI_DEV_INDEX_STORE_TRANSCRIPTS=true` keeps transcripts in your local SQLite DB for private full-text search.
- If you ever expose an instance publicly, set `AI_DEV_INDEX_STORE_TRANSCRIPTS=false` so transcripts are processed ephemerally and only original summaries, extracted topics, and links back to the creators' YouTube videos are retained.
