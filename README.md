# Browser AI Agent

A comprehensive AI-powered browser automation framework built on Chrome DevTools Protocol (CDP) and Playwright. It enables LLM-driven agents to interact with web pages through a structured action system, supporting vision-based reasoning, DOM analysis, multi-tab navigation, file management, skill execution, and more.

## Quick Start

### Docker (recommended)

```bash
# Copy .env.example to .env and set your API keys
cp .env.example .env

# Build and run
docker compose up --build

# Open http://localhost:8765
```

### Manual

```bash
# Install dependencies
pip install -r requirements.txt

# Copy .env.example to .env and set your API keys
cp .env.example .env

# Install Playwright browsers
playwright install --with-deps chromium

# Start the web interface
python3 web_ui.py
# Open http://localhost:8765
```

## Features

- **Action Graph Engine** — Dependency graph for sub-goals with branching, fallback strategies, and condition logic
- **Multi-browser support** — Chromium (CDP), Firefox, WebKit (Playwright)
- **Multi-LLM support** — OpenAI, Google Gemini, Anthropic Claude, Mistral, Cerebras, GLM, Ollama, Azure, OCI
- **Vision + DOM** — Screenshot-based reasoning combined with DOM element interaction
- **Action system** — Search, navigate, click, type, scroll, extract data, execute JavaScript, manage files
- **Event-driven architecture** — Watchdog-based monitoring (crashes, popups, downloads, screenshots, security)
- **Smart recovery** — Condition handlers (captcha → rotate session, blocked → proxy switch, timeout → retry)
- **Session rotation** — Automatic geo-proxy rotation and GoLogin profile switching on failures
- **Profile management** — Named profiles with proxy, headless mode, persistent sessions
- **Task queue** — Queue multiple tasks, run sequentially
- **Memory system** — SQLite-based persistent memory across tasks with cross-task context recall
- **Security** — Domain allowlists/blocklists, destructive action confirmation
- **Skill system** — Reusable API-driven automation workflows
- **Demo mode** — In-browser log panel for real-time agent activity
- **Video recording** — MP4 session recording from CDP screencast
- **GIF export** — Animated GIF from execution history with step overlays
- **Token tracking** — Per-model cost estimation with LiteLLM pricing
- **File system** — In-memory + disk file management (MD, TXT, JSON, CSV, PDF, DOCX)

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Action Graph Engine](#action-graph-engine)
- [Module Reference](#module-reference)
  - [Orchestrator](#orchestrator)
  - [Agent](#agent)
  - [Browser](#browser)
  - [Tools](#tools)
  - [DOM](#dom)
  - [LLM](#llm)
  - [Skills](#skills)
  - [Tokens](#tokens)
  - [Telemetry](#telemetry)
  - [Screenshots](#screenshots)
  - [Filesystem](#filesystem)
  - [Config](#config)
  - [Observability](#observability)
  - [Sync](#sync)
- [Applications](#applications)
  - [Task Hunter](#task-hunter)
  - [Web UI](#web-ui)
- [Comparison Table](#comparison-table)
- [Supported LLM Providers](#supported-llm-providers)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      User API                           │
│  web_ui.py (FastAPI) ─► REST endpoints + static UI      │
├─────────────────────────────────────────────────────────┤
│              Orchestrator Layer + Action Graph Engine    │
│  TaskOrchestrator (FSM: idle→plan→exec→validate→done)   │
│    │                                                    │
│    ├── Structured Planner (LLM → dependency graph)      │
│    ├── Graph Executor (topo sort → ready goals → run)   │
│    ├── Condition Handlers (captcha/blocked/timeout)      │
│    ├── Failure Dispatcher (fallback/skip/retry/rotate)  │
│    ├── Session Rotator (geo-proxy + GoLogin cycling)    │
│    ├── Memory System (SQLite: cross-task persistence)   │
│    └── Validator (failure classification + feedback)     │
├─────────────────────────────────────────────────────────┤
│                      Agent Layer                        │
│  Agent ─► MessageManager ─► SystemPrompt ─► LLM        │
│    │          │                                         │
│    ├── Judge (trace evaluation)                         │
│    ├── GIF Generator (visual history)                   │
│    └── VariableDetector (rerun support)                 │
├─────────────────────────────────────────────────────────┤
│                     Tools Layer                         │
│  Tools/Controller ─► Registry ─► ActionModel            │
│  (click, input, scroll, navigate, extract, evaluate…)   │
├─────────────────────────────────────────────────────────┤
│                    Browser Layer                        │
│  BrowserSession ─► SessionManager ─► CDP/Playwright     │
│    │                                                    │
│    ├── EventBus (bubus) ─► Watchdogs                    │
│    ├── DemoMode (in-browser log panel)                  │
│    ├── VideoRecorder (MP4 sessions)                     │
│    └── BrowserProfile (launch config)                   │
├─────────────────────────────────────────────────────────┤
│                      DOM Layer                          │
│  DomService ─► EnhancedSnapshot ─► Serializer           │
│  MarkdownExtractor (page content → clean markdown)      │
├─────────────────────────────────────────────────────────┤
│                   Support Layers                        │
│  LLM (OpenAI, Google, Azure, Anthropic, Cerebras, …)   │
│  Skills (API-driven reusable workflows)                 │
│  Tokens (cost tracking via LiteLLM pricing)             │
│  Telemetry (anonymous usage via PostHog)                │
│  Screenshots (disk-based storage)                       │
│  FileSystem (in-memory + disk file management)          │
│  Sync (cloud auth for remote browser sessions)          │
│  Observability (structured tracing & span tracking)     │
│  Config (env + JSON config with migration)              │
└─────────────────────────────────────────────────────────┘
```

---

## Action Graph Engine

The Action Graph Engine is the core orchestration layer that transforms a linear task plan into an executable dependency graph. This is what differentiates a professional agent from a simple one.

### How It Works

```
User Task: "Find best restaurants in Moscow and save results"
                    │
                    ▼
            ┌── Planner (LLM) ──┐
            │                    │
            ▼                    ▼
     ┌─────────────┐    ┌──────────────┐
     │ #1 Search on │    │ #2 Search on │
     │  DuckDuckGo  │───►│   Google     │
     │              │    │  (fallback)  │
     └──────┬───────┘    └──────────────┘
            │ depends_on
            ▼
     ┌─────────────┐
     │ #3 Click     │
     │ first result │
     └──────┬───────┘
            │ depends_on
            ▼
     ┌─────────────┐
     │ #4 Extract   │
     │ top 3 items  │
     └─────────────┘
```

### SubGoal Structure

Each sub-goal is a node in the dependency graph:

```python
@dataclass
class SubGoal:
    id: int
    description: str
    status: str             # pending | executing | done | failed | skipped
    depends_on: list[int]   # IDs of sub-goals that must complete first
    fallback_goal_id: int   # Alternative sub-goal if this one fails
    on_failure: str         # fail | skip | fallback
    on_captcha: str         # fail | fallback | rotate_session
    on_timeout: str         # retry | fail | skip
```

### Graph Execution

Instead of a linear `for` loop, the engine uses topological sorting and ready-goal detection:

1. **Topological Sort** — Orders sub-goals by dependencies
2. **Ready Detection** — Finds goals whose dependencies are all `done` or `skipped`
3. **Execute** — Runs the first ready goal through the Agent executor
4. **Validate** — LLM validator classifies the result (success/failure type)
5. **Dispatch** — Condition handlers decide next action based on failure type
6. **Repeat** — Until all goals are resolved or the graph is stuck

### Condition Handlers

The engine dispatches failures to specialized handlers:

| Condition | Handler | Action |
|-----------|---------|--------|
| `captcha` | `handle_captcha()` | Rotate session + retry |
| `blocked` | `handle_blocked()` | Switch proxy + retry |
| `timeout` | `handle_timeout()` | Retry with same config |
| `login_required` | `handle_login_required()` | Skip (can't auto-login) |

### Failure Dispatch Flow

```
Validator returns failure
        │
        ▼
1. Try condition handler for failure_type
        │ (captcha → rotate, blocked → proxy switch)
        ▼
2. Use recommended_action from validator
        │ (retry, fallback, rotate_session, skip, abort)
        ▼
3. Use sub-goal's own on_failure strategy
        │ (fail, skip, fallback)
        ▼
4. Default: fail
```

### Structured Validator

The validator now returns detailed failure classification:

```json
{
  "success": false,
  "failure_type": "captcha",
  "should_retry": false,
  "recommended_action": "rotate_session",
  "reason": "Page shows reCAPTCHA challenge"
}
```

Supported failure types: `captcha`, `timeout`, `blocked`, `not_found`, `login_required`, `crash`

### Session Rotation

When a failure requires session rotation, the engine:

1. Switches to the next geo-proxy country from the pool
2. Restarts the GoLogin profile (if configured)
3. Retries the failed sub-goal with the new session

### Planner Output

The LLM planner generates a structured graph:

```json
[
  {"id": 0, "description": "Search on DuckDuckGo", "depends_on": [], "on_failure": "fallback", "fallback_goal_id": 1},
  {"id": 1, "description": "Search on Google (fallback)", "depends_on": [], "on_failure": "fail"},
  {"id": 2, "description": "Click first result", "depends_on": [0], "on_failure": "fail"},
  {"id": 3, "description": "Extract data", "depends_on": [2], "on_failure": "fail"}
]
```

---

## Module Reference

### Orchestrator

#### `orchestrator.py` — `TaskOrchestrator`

The middle layer between User API and the Executor (Agent). Implements a finite state machine with an Action Graph Engine that coordinates planning, graph execution, validation, condition handling, and memory across task sub-goals.

- **Main classes:**
  - `TaskOrchestrator` — FSM engine with states: `idle` → `planning` → `executing` → `validating` → `done` / `failed`
  - `OrchestratorConfig` — Configuration: planner, validator, memory, geo, GoLogin, retries, max steps
  - `OrchestratorContext` — Runtime context: task state, plan, sub-goals, memory entries, timing
  - `SubGoal` — Graph node with dependency tracking (`depends_on`), fallback routing (`fallback_goal_id`), and failure strategies (`on_failure`, `on_captcha`, `on_timeout`)
  - `ExecutorResult` — Structured result from executor with `error_type` classification
  - `MemorySystem` — SQLite-based persistent memory for cross-task context (thread-safe)
- **Key features:**
  - **Action Graph Engine:** Dependency graph with topological sorting, ready-goal detection, and cycle handling
  - **Structured Planner:** LLM generates JSON graph of sub-goals with `depends_on` and `fallback_goal_id`
  - **Condition Handlers:** `handle_captcha`, `handle_blocked`, `handle_timeout`, `handle_login_required` — automatic failure response
  - **Failure Dispatcher:** Multi-layer dispatch: condition handler → validator recommendation → sub-goal strategy
  - **Session Rotation:** `rotate_browser_session()` — cycles geo-proxy countries and restarts GoLogin profiles
  - **Validator with failure classification:** Returns `failure_type`, `recommended_action`, and `reason`
  - **Memory System:** SQLite DB stores task results, sub-goal outcomes, and cross-task context; recalled during planning
  - **Geo-proxy:** Country-based proxy selection via env vars (`GEO_PROXY_US`, etc.) or presets
  - **GoLogin integration:** Starts GoLogin profile via API, gets CDP URL, passes to `BrowserSession`
  - **Backward compatible:** When orchestrator features are disabled, falls back to direct `run_task()` execution

- **Graph utilities:**
  - `topo_sort(goals)` — Topological sort by `depends_on`; falls back to original order on cycle detection
  - `get_ready_goals(goals)` — Returns all pending goals whose dependencies are satisfied
  - `dispatch_failure(...)` — Multi-layer failure routing to appropriate action

---

### Agent

#### `browser_ai/agent/service.py` — `Agent`

The core orchestrator that drives the browser automation loop. It coordinates between the LLM, browser session, tools, and message manager to execute multi-step tasks autonomously.

- **Main class:** `Agent[Context, AgentStructuredOutput]`
- **Key methods:**
  - `run(max_steps)` — Main entry point. Runs the agent loop until the task completes or max steps is reached.
  - `step(step_info)` — Executes a single agent step: get browser state → build messages → call LLM → execute actions.
  - `pause()` / `resume()` / `stop()` — Control the agent execution lifecycle.
  - `rerun(history)` — Re-execute a previous agent trace (for replay/testing).
  - `get_history()` — Returns the complete `AgentHistoryList` with all actions, results, and metadata.
- **Key features:**
  - Automatic LLM timeout configuration per model family (Gemini, Claude, GPT, etc.)
  - Coordinate-based clicking auto-enabled for Claude Sonnet 4.5+, Gemini 3 Pro, and browser-use models
  - Flash mode for lightweight/fast execution (disables thinking and evaluation)
  - Judge integration for post-execution trace validation
  - GIF generation from execution history
  - Skills integration (API-driven reusable workflows)
  - Structured output support via Pydantic models
  - Callbacks for step progress and done events
  - Follow-up task support (continue from previous session)
  - Sensitive data masking throughout the message pipeline
  - Demo mode (streams logs to an in-browser panel)

#### `browser_ai/agent/views.py` — Agent Data Models

Defines all Pydantic data models used throughout the agent system for configuration, state, output, and history.

- **Main classes:**
  - `AgentSettings` — Configuration (vision, max failures, flash mode, judge, timeouts, etc.)
  - `AgentState` — Mutable runtime state (step counter, pause/resume, failures, messages)
  - `AgentOutput` — LLM response structure (thinking, evaluation, memory, next_goal, actions)
  - `AgentBrain` — The agent's cognitive state per step
  - `ActionResult` — Result of executing a single action (done, success, error, extracted content, images, metadata)
  - `AgentHistory` — Single step history (model output + results + browser state + metadata)
  - `AgentHistoryList` — Complete execution trace with serialization, filtering, and analysis methods
  - `JudgementResult` — LLM judge verdict (reasoning, verdict, failure reason, impossible task, captcha detection)
  - `StepMetadata` — Timing information per step
  - `DetectedVariable` / `VariableMetadata` — For variable detection in reruns

#### `browser_ai/agent/prompts.py` — `SystemPrompt` / `AgentMessagePrompt`

Manages system and user prompts for the agent, including dynamic prompt template selection based on model type and mode.

- **Main classes:**
  - `SystemPrompt` — Loads and formats system prompt templates (thinking/no-thinking/flash/anthropic/browser-use variants)
  - `AgentMessagePrompt` — Builds the per-step user message with browser state, agent history, file system state, screenshots, page statistics, and available actions

#### `browser_ai/agent/message_manager/service.py` — `MessageManager`

Manages the conversation message history between the agent and the LLM.

#### `browser_ai/agent/judge.py` — Judge System

Constructs prompts for an LLM judge that evaluates agent execution traces for success/failure.

#### `browser_ai/agent/gif.py` — GIF Generator

Creates animated GIF recordings from agent execution history with overlaid task descriptions and step goals.

#### `browser_ai/agent/variable_detector.py` — Variable Detector

Analyzes agent execution history to detect reusable variables for replay/rerun scenarios.

---

### Browser

#### `browser_ai/browser/session.py` — `BrowserSession`

The core browser abstraction providing a 2-layer architecture: high-level event handling for agents and direct CDP/Playwright calls for operations.

- **Main classes:**
  - `BrowserSession` — Full browser lifecycle management with event bus, session manager, and watchdog system
  - `Target` — Browser target (page, iframe, worker) with URL and title
  - `CDPSession` — CDP communication channel to a target
- **Key features:**
  - Event-driven architecture via `bubus` EventBus
  - Automatic CDP session recovery on target crash/detach
  - Cloud browser support (browser-use cloud service)
  - Cross-origin iframe support (configurable depth and limits)
  - Storage state management (cookies, localStorage, IndexedDB)
  - Proxy authentication handling
  - Download tracking

#### `browser_ai/browser/driver.py` — `PlaywrightDriver`

Cross-browser driver abstraction enabling Chromium, Firefox, and WebKit support through Playwright.

#### `browser_ai/browser/profile.py` — `BrowserProfile`

Comprehensive browser configuration model that covers all Playwright launch, context, and connection parameters.

#### `browser_ai/browser/events.py` — Browser Events

Defines the complete event vocabulary for browser communication.

#### `browser_ai/browser/session_manager.py` — `SessionManager`

Event-driven CDP session manager that automatically synchronizes the session pool with browser state.

#### `browser_ai/browser/demo_mode.py` — `DemoMode`

Injects and manages an in-browser log panel that streams agent activity in real-time.

#### `browser_ai/browser/video_recorder.py` — `VideoRecorderService`

Records browser sessions as MP4 videos from CDP screencast frames.

#### `browser_ai/browser/watchdog_base.py` — `BaseWatchdog`

Base class for all browser monitoring components (watchdogs).

#### `browser_ai/browser/watchdogs/` — Built-in Watchdogs

| Watchdog | Purpose |
|----------|---------|
| `screenshot_watchdog.py` | Captures screenshots on browser state request events |
| `dom_watchdog.py` | Builds DOM tree on state request, manages serialization |
| `crash_watchdog.py` | Detects target crashes, triggers session recovery |
| `popups_watchdog.py` | Auto-handles popups, dialogs, permission prompts |
| `permissions_watchdog.py` | Manages browser permission grants/denials |
| `downloads_watchdog.py` | Tracks file downloads, manages download paths |
| `recording_watchdog.py` | Drives video recording via `VideoRecorderService` |
| `aboutblank_watchdog.py` | Handles `about:blank` pages (initial load state) |
| `default_action_watchdog.py` | Executes default browser actions (click, type, scroll, navigate) |
| `local_browser_watchdog.py` | Manages local browser launch and lifecycle |
| `security_watchdog.py` | Enforces domain allowlists/blocklists |
| `storage_state_watchdog.py` | Saves/loads cookies, localStorage, sessionStorage between runs |

---

### Tools

#### `browser_ai/tools/service.py` — `Tools` / `Controller`

The action registry and execution engine that provides all browser automation actions available to the agent.

- **Registered actions:**
  - **Navigation:** `search` (DuckDuckGo/Google/Bing), `navigate`, `go_back`, `wait`
  - **Interaction:** `click` (index or coordinate), `input` (text entry), `send_keys`, `scroll`, `find_text`
  - **Extraction:** `extract` (LLM-powered page content extraction with markdown)
  - **Dropdowns:** `dropdown_options`, `select_dropdown`
  - **Tabs:** `switch`, `close`
  - **Files:** `write_file`, `read_file`, `replace_file`, `upload_file`
  - **Meta:** `screenshot`, `evaluate` (JavaScript execution), `done` (task completion)

---

### DOM

#### `browser_ai/dom/service.py` — `DomService`

Service for extracting and processing the DOM tree from browser pages using CDP snapshot, accessibility tree, and layout data.

#### `browser_ai/dom/views.py` — DOM Data Models

Core data structures for the DOM tree representation.

#### `browser_ai/dom/enhanced_snapshot.py` — Enhanced Snapshot Processing

Stateless functions for parsing Chrome DevTools Protocol DOMSnapshot data.

#### `browser_ai/dom/markdown_extractor.py` — Markdown Extractor

Unified interface for extracting clean markdown from browser content.

---

### LLM

#### `browser_ai/llm/base.py` — `BaseChatModel`

Protocol definition for all LLM chat model implementations.

#### `browser_ai/llm/models.py` — Model Registry

Convenient lazy-loaded access to all supported LLM models with auto-configuration from environment variables.

---

### Skills

#### `browser_ai/skills/service.py` — `SkillService`

Service for fetching, caching, and executing reusable automation skills from the Browser Use API.

---

### Tokens

#### `browser_ai/tokens/service.py` — `TokenCost`

Token usage tracking and cost calculation service.

---

### Telemetry

#### `browser_ai/telemetry/service.py` — `ProductTelemetry`

Anonymized telemetry service using PostHog for usage analytics. Fully opt-out via `ANONYMIZED_TELEMETRY=False`.

---

### Screenshots

#### `browser_ai/screenshots/service.py` — `ScreenshotService`

Simple disk-based screenshot storage service.

---

### Filesystem

#### `browser_ai/filesystem/file_system.py` — `FileSystem`

In-memory file system with disk persistence supporting multiple file types (MD, TXT, JSON, CSV, PDF, DOCX).

---

### Config

#### `browser_ai/config.py` — Configuration System

Unified configuration system with environment variables, JSON config file, and automatic migration support.

---

### Observability

#### `browser_ai/observability.py`

Observability decorators providing optional integration with Laminar (lmnr) for distributed tracing.

---

### Sync

#### `browser_ai/sync/auth.py`

OAuth2 Device Authorization Grant flow client for authenticating with the cloud browser service.

---

## Applications

### Task Hunter

#### `task_hunter.py`

Core task orchestration module for running browser automation tasks with profile management, task queuing, and multi-provider LLM support.

- **Key features:**
  - **Profile management:** Create, update, delete, and switch between named configurations
  - **Task queue:** Add tasks to a queue, run them sequentially with the selected profile
  - **Multi-provider support:** Custom (browser-use), OpenAI, GLM
  - **Proxy support:** Server, bypass, username, password per profile
  - **Custom tools:** `wait_for_user` (manual step confirmation), `confirm_destructive` (approval before destructive actions)
  - **Screenshot capture:** Saves step screenshots to `artifacts/screenshots/`
  - **Stuck detection:** Repeating action threshold to break loops
  - **Fallback LLM:** Optional secondary model when primary fails
  - **Sub-agent planning:** Optional multi-agent task decomposition
  - **Cross-browser:** CDP (Chromium) or Playwright (Chromium/Firefox/WebKit)
  - **Persistent sessions:** Reuse browser profiles across runs

### Web UI

#### `web_ui.py`

FastAPI-based web interface for Task Hunter with REST API endpoints and a static frontend.

- **API endpoints:**
  - `GET /api/profiles` — List all profiles
  - `GET /api/profiles/{name}` — Get profile details
  - `POST /api/run` — Start a task (returns task_id for polling)
  - `GET /api/run/{task_id}` — Poll task status, steps, and graph state
  - `POST /api/run/{task_id}/confirm` — Answer a pending confirmation question
  - `GET /api/queue` — Get task queue
  - `POST /api/queue` — Add task to queue
  - `GET /api/memory/recent` — Recent memory entries
  - `GET /api/memory/tasks` — Task execution history
  - `GET /api/memory/search` — Search memory by key/category
  - `GET /api/geo/countries` — Available geo-proxy countries
- **Key features:**
  - CORS enabled for any origin
  - In-memory run status tracking with step-by-step progress
  - Orchestrator state broadcasting (plan graph, sub-goal statuses, current execution)
  - Confirmation flow (agent asks question → UI shows it → user responds)
  - Browser backend selection (CDP/Playwright) per run
  - Safe I/O handling (global BrokenPipeError protection)

---

## Comparison Table

| Criteria | Simple Agent | Professional Agent | This Project |
|----------|-------------|-------------------|-------------|
| One LLM | Planner + Executor + Validator | **All 3 present** |
| No memory | Multi-layer memory | **SQLite persistent memory** |
| One browser | Browser cluster | Session rotation (geo-proxy + GoLogin) |
| `sleep()` | Event-driven waits | **EventBus + 12 watchdogs** |
| No retry | Smart recovery | **Condition handlers + fallback + retry** |
| No logging | Full observability | **Tracing, telemetry, tokens, demo, video, GIF, screenshots** |
| No graph | Action Graph Engine | **Dependency graph + topo sort + fallback + conditions** |

---

## Supported LLM Providers

| Provider | Models | Config Env Var |
|----------|--------|----------------|
| OpenAI | GPT-4o, GPT-4.1-mini, o1/o3/o4 series, GPT-5 | `OPENAI_API_KEY` |
| Azure OpenAI | Same as OpenAI | `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Google | Gemini 2.0/2.5 (Flash, Pro, Flash-Lite) | `GOOGLE_API_KEY` |
| Anthropic | Claude (via OpenAI-compatible) | `ANTHROPIC_API_KEY` |
| Mistral | Large, Medium, Small, Codestral, Pixtral | `MISTRAL_API_KEY` |
| Cerebras | Llama 3.1/3.3/4, GPT-OSS, Qwen 3 | `CEREBRAS_API_KEY` |
| Browser Use | bu-latest, bu-1.0 | `BROWSER_USE_API_KEY` |
| OCI (optional) | Via ChatOCIRaw | Manual config |

---

## Usage Examples

### Programmatic (Python)

```python
import asyncio
from browser_ai import Agent, Browser

async def main():
    agent = Agent(
        task="Search for the latest AI news and summarize the top 3 results",
        browser=Browser(),
    )
    history = await agent.run(max_steps=20)
    print(history.final_result())

asyncio.run(main())
```

### With Orchestrator (dependency graph + fallback)

```python
import asyncio
from orchestrator import TaskOrchestrator, OrchestratorConfig, MemorySystem

async def main():
    memory = MemorySystem()
    orchestrator = TaskOrchestrator(memory=memory)

    config = OrchestratorConfig(
        enable_planner=True,       # LLM decomposes task into sub-goal graph
        enable_validator=True,     # LLM validates each step with failure classification
        enable_memory=True,        # Cross-task memory
        geo_country="us",          # Geo-proxy
        max_retries_per_subgoal=2, # Retry on failure
    )

    ctx = await orchestrator.run(
        task="Find best restaurants in Moscow and extract top 5",
        profile_config={"provider": "custom"},
        config=config,
    )

    print(f"State: {ctx.state}")
    for g in ctx.plan:
        print(f"  #{g.id}: {g.description} [{g.status}]")

asyncio.run(main())
```

### Web UI

```bash
python3 web_ui.py
# Open http://localhost:8765 — select LLM provider, browser backend, and run tasks via the UI
```

---

## Project Structure

```
browser_ai_agent/
├── web_ui.py                    # FastAPI web server + REST API
├── orchestrator.py              # Action Graph Engine + FSM orchestrator
├── task_hunter.py               # Task execution, profiles, queue
├── static/index.html            # Web UI frontend (dependency graph visualization)
├── .env.example                 # API keys template
├── requirements-web.txt         # Dependencies
└── browser_ai/                  # Core library
    ├── agent/                   # AI agent orchestration
    ├── browser/                 # Browser session + drivers + watchdogs
    ├── tools/                   # Action registry (click, type, navigate...)
    ├── dom/                     # DOM processing + serialization
    ├── llm/                     # LLM providers (15+ integrations)
    ├── skills/                  # Reusable automation workflows
    ├── tokens/                  # Token counting + cost estimation
    ├── telemetry/               # Anonymous usage analytics
    ├── screenshots/             # Screenshot storage
    ├── filesystem/              # In-memory + disk file system
    ├── sync/                    # Cloud authentication
    ├── config.py                # Env-based configuration
    ├── observability.py         # Tracing decorators (Laminar)
    └── logging_config.py        # Structured logging
```
