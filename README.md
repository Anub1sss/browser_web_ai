# Browser AI Agent

A comprehensive AI-powered browser automation framework built on Chrome DevTools Protocol (CDP) and Playwright. It enables LLM-driven agents to interact with web pages through a structured action system, supporting vision-based reasoning, DOM analysis, multi-tab navigation, file management, skill execution, and more.

## Quick Start

```bash
# Install dependencies
pip install -r requirements-web.txt

# Copy .env.example to .env and set your API keys
cp .env.example .env

# Install Playwright browsers (needed for Firefox/WebKit)
python3 -m playwright install

# Start the web interface
python3 web_ui.py
# Open http://localhost:8765
```

## Features

- **Multi-browser support** — Chromium (CDP), Firefox, WebKit (Playwright)
- **Multi-LLM support** — OpenAI, Google Gemini, Anthropic Claude, Mistral, Cerebras, GLM, Ollama, Azure, OCI
- **Vision + DOM** — Screenshot-based reasoning combined with DOM element interaction
- **Action system** — Search, navigate, click, type, scroll, extract data, execute JavaScript, manage files
- **Event-driven architecture** — Watchdog-based monitoring (crashes, popups, downloads, screenshots, security)
- **Profile management** — Named profiles with proxy, headless mode, persistent sessions
- **Task queue** — Queue multiple tasks, run sequentially
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
- [Module Reference](#module-reference)
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
- [Supported LLM Providers](#supported-llm-providers)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
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

## Module Reference

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
- **Key features:**
  - Sensitive data filtering in serialization
  - Structured output parsing from history
  - Screenshot path/base64 extraction from history
  - Action history formatting for judge evaluation

#### `browser_ai/agent/prompts.py` — `SystemPrompt` / `AgentMessagePrompt`

Manages system and user prompts for the agent, including dynamic prompt template selection based on model type and mode.

- **Main classes:**
  - `SystemPrompt` — Loads and formats system prompt templates (thinking/no-thinking/flash/anthropic/browser-use variants)
  - `AgentMessagePrompt` — Builds the per-step user message with browser state, agent history, file system state, screenshots, page statistics, and available actions
- **Key features:**
  - 8 different system prompt templates for different model types and modes
  - Page statistics extraction (links, iframes, shadow DOM, scroll containers, interactive elements)
  - Screenshot resizing for LLM-optimized dimensions
  - PDF viewer detection with appropriate instructions
  - Sensitive data placeholder injection
  - Rerun summary prompt generation

#### `browser_ai/agent/message_manager/service.py` — `MessageManager`

Manages the conversation message history between the agent and the LLM, handling message construction, history truncation, and sensitive data filtering.

- **Main class:** `MessageManager`
- **Key methods:**
  - `create_state_messages()` — Builds the current browser state message with DOM, screenshots, file system, and action history
  - `get_messages()` — Returns the final message list ready for LLM invocation
  - `add_new_task(task)` — Appends a follow-up task to the conversation
- **Key features:**
  - Configurable `max_history_items` with intelligent omission (keeps first + most recent)
  - Read-state management (one-time content vs persistent memory)
  - Sensitive data filtering across all message types
  - Context message injection (validation errors, retries, timeouts)
  - Domain-based sensitive data scoping

#### `browser_ai/agent/judge.py` — Judge System

Constructs prompts for an LLM judge that evaluates agent execution traces for success/failure.

- **Main function:** `construct_judge_messages(task, final_result, agent_steps, screenshot_paths, ...)`
- **Key features:**
  - Detailed evaluation framework with 5 criteria (task satisfaction, output quality, tool effectiveness, reasoning, browser handling)
  - Ground truth validation (highest priority, overrides all other criteria)
  - Impossible task detection (vague instructions, broken sites, missing credentials)
  - CAPTCHA detection
  - Screenshot evidence (up to 10 screenshots per evaluation)
  - Structured JSON output (reasoning, verdict, failure reason, impossible task, captcha flags)

#### `browser_ai/agent/gif.py` — GIF Generator

Creates animated GIF recordings from agent execution history with overlaid task descriptions and step goals.

- **Main function:** `create_history_gif(task, history, output_path, ...)`
- **Key features:**
  - Task title frame with dynamic font sizing
  - Per-step goal overlays with step numbers
  - Multi-font support (CJK, Arabic, Latin — PingFang, STHeiti, Noto Sans CJK, etc.)
  - Placeholder screenshot filtering (skips about:blank and new tab pages)
  - Unicode escape sequence handling for international text
  - Optional logo overlay

#### `browser_ai/agent/variable_detector.py` — Variable Detector

Analyzes agent execution history to detect reusable variables for replay/rerun scenarios.

- **Main function:** `detect_variables_in_history(history) -> dict[str, DetectedVariable]`
- **Key features:**
  - Two detection strategies: element attributes (HTML id, name, type, placeholder) and value pattern matching
  - Detects: email, phone, date, name (first/last/full), number, address, zip code, company, city, state, country
  - Automatic unique naming with numeric suffixes for duplicates
  - Context-aware detection using interacted DOM element attributes

---

### Browser

#### `browser_ai/browser/session.py` — `BrowserSession`

The core browser abstraction providing a 2-layer architecture: high-level event handling for agents and direct CDP/Playwright calls for operations.

- **Main classes:**
  - `BrowserSession` — Full browser lifecycle management with event bus, session manager, and watchdog system
  - `Target` — Browser target (page, iframe, worker) with URL and title
  - `CDPSession` — CDP communication channel to a target
- **Key methods:**
  - `start()` / `stop()` — Browser lifecycle (launch, connect via CDP or cloud)
  - `get_or_create_cdp_session()` — Session pool management with recovery
  - `get_state()` — Capture current browser state (DOM + screenshot + tabs)
  - `get_element_by_index()` / `get_selector_map()` — DOM element lookup
  - `highlight_interaction_element()` / `highlight_coordinate_click()` — Visual feedback
- **Key features:**
  - Event-driven architecture via `bubus` EventBus
  - Automatic CDP session recovery on target crash/detach
  - Cloud browser support (browser-use cloud service)
  - Cross-origin iframe support (configurable depth and limits)
  - Storage state management (cookies, localStorage, IndexedDB)
  - Proxy authentication handling
  - Download tracking

#### `browser_ai/browser/driver.py` — `PlaywrightDriver`

Cross-browser driver abstraction enabling Chromium, Firefox, and WebKit support through Playwright while maintaining the same DOM recognition interface.

- **Main class:** `PlaywrightDriver`
- **Key features:**
  - Alternative to CDP for Firefox/WebKit support
  - Builds minimal DOM tree from Playwright's API
  - Handles navigation, clicking, typing, scrolling, and screenshots
  - Tab management via Playwright pages
  - Viewport configuration

#### `browser_ai/browser/profile.py` — `BrowserProfile`

Comprehensive browser configuration model that covers all Playwright launch, context, and connection parameters plus custom automation-specific settings.

- **Main classes:**
  - `BrowserProfile` — Master configuration combining launch args, context args, connect args, and custom options
  - `ProxySettings` — Typed proxy configuration (server, bypass, username, password)
  - `BrowserLaunchArgs` / `BrowserContextArgs` / `BrowserConnectArgs` — Layered Playwright parameter models
- **Key features:**
  - Default Chrome extensions (uBlock Origin, cookie handler, ClearURLs, Force Background Tab) with auto-download
  - Domain allowlisting/blocklisting with auto-optimization for large lists (100+ items → set for O(1) lookup)
  - IP address blocking
  - Display auto-detection (macOS AppKit, Windows/Linux screeninfo)
  - Deterministic rendering mode (same screenshots across OS)
  - Chrome profile copying to temp directories for parallel runs
  - Cross-browser backend selection (`cdp` or `playwright`)
  - Demo mode side panel configuration
  - Video recording configuration (directory, size, framerate)
  - HAR recording support
  - Extensive Chrome CLI arg management with deduplication and feature merging

#### `browser_ai/browser/events.py` — Browser Events

Defines the complete event vocabulary for browser communication, covering navigation, interaction, lifecycle, and error events.

- **Action events:** `NavigateToUrlEvent`, `ClickElementEvent`, `ClickCoordinateEvent`, `TypeTextEvent`, `ScrollEvent`, `ScrollToTextEvent`, `SendKeysEvent`, `UploadFileEvent`, `GetDropdownOptionsEvent`, `SelectDropdownOptionEvent`
- **Tab events:** `SwitchTabEvent`, `CloseTabEvent`, `TabCreatedEvent`, `TabClosedEvent`
- **Navigation events:** `GoBackEvent`, `GoForwardEvent`, `RefreshEvent`, `WaitEvent`
- **Browser lifecycle:** `BrowserStartEvent`, `BrowserStopEvent`, `BrowserLaunchEvent`, `BrowserKillEvent`, `BrowserConnectedEvent`, `BrowserStoppedEvent`
- **State events:** `BrowserStateRequestEvent`, `ScreenshotEvent`, `AgentFocusChangedEvent`
- **Storage events:** `SaveStorageStateEvent`, `LoadStorageStateEvent`, `StorageStateSavedEvent`, `StorageStateLoadedEvent`
- **Other:** `FileDownloadedEvent`, `DialogOpenedEvent`, `TargetCrashedEvent`, `BrowserErrorEvent`
- **Key features:**
  - All events have configurable timeouts via environment variables
  - Event name uniqueness validation at import time (prevents substring collisions)
  - Type-safe result types per event via generics

#### `browser_ai/browser/session_manager.py` — `SessionManager`

Event-driven CDP session manager that automatically synchronizes the session pool with browser state via Chrome target attach/detach events.

- **Main class:** `SessionManager`
- **Key features:**
  - Single source of truth for all targets and sessions
  - Automatic session creation/removal via CDP events
  - Agent focus recovery when target crashes/detaches (creates emergency fallback tabs)
  - Event-driven coordination instead of polling
  - Target info updates via `targetInfoChanged` events
  - Page lifecycle monitoring (load, DOMContentLoaded, networkIdle)
  - Concurrent operation support with asyncio locks

#### `browser_ai/browser/demo_mode.py` — `DemoMode`

Injects and manages an in-browser log panel that streams agent activity in real-time, useful for demos and debugging.

- **Main class:** `DemoMode`
- **Key methods:**
  - `ensure_ready()` — Inject panel script into all open pages
  - `send_log(message, level, metadata)` — Send log entry to the panel
- **Key features:**
  - Beautiful dark-themed side panel with expand/collapse per entry
  - Log levels: info, action, thought, success, warning, error
  - Persists state across navigation via sessionStorage
  - Auto-injection into new pages via CDP init script
  - Responsive layout with adaptive panel width
  - Session ID isolation for multi-agent scenarios

#### `browser_ai/browser/video_recorder.py` — `VideoRecorderService`

Records browser sessions as MP4 videos from CDP screencast frames.

- **Main class:** `VideoRecorderService`
- **Key methods:**
  - `start()` — Initialize video writer with codec settings
  - `add_frame(frame_data_b64)` — Decode, resize, pad, and append a frame
  - `stop_and_save()` — Finalize and close the video file
- **Key features:**
  - Uses `imageio` + `ffmpeg` backend (pip-installable)
  - Automatic frame resizing and macro-block padding for codec compatibility
  - H.264 codec with yuv420p pixel format for universal playback

#### `browser_ai/browser/watchdog_base.py` — `BaseWatchdog`

Base class for all browser monitoring components (watchdogs) that react to browser events.

- **Main class:** `BaseWatchdog`
- **Key features:**
  - Automatic event handler registration based on method naming convention (`on_EventName`)
  - `LISTENS_TO` / `EMITS` class variables for documentation and validation
  - Unique handler naming to prevent duplicate registration
  - Automatic CDP session repair on handler errors
  - Task cleanup during garbage collection
  - Debug logging with timing, parent event chain, and result tracking

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
| `security_watchdog.py` | Enforces domain allowlists/blocklists, blocks navigation to restricted URLs |
| `storage_state_watchdog.py` | Saves/loads cookies, localStorage, sessionStorage between runs |

---

### Tools

#### `browser_ai/tools/service.py` — `Tools` / `Controller`

The action registry and execution engine that provides all browser automation actions available to the agent.

- **Main classes:**
  - `Tools[Context]` — Default tool set with all browser actions
  - `CodeAgentTools[Context]` — Optimized for Python-based code agents (excludes extract, find_text, screenshot, search, file system actions)
  - `Controller` — Backward compatibility alias for `Tools`
- **Registered actions:**
  - **Navigation:** `search` (DuckDuckGo/Google/Bing), `navigate`, `go_back`, `wait`
  - **Interaction:** `click` (index or coordinate), `input` (text entry), `send_keys`, `scroll`, `find_text`
  - **Extraction:** `extract` (LLM-powered page content extraction with markdown)
  - **Dropdowns:** `dropdown_options`, `select_dropdown`
  - **Tabs:** `switch`, `close`
  - **Files:** `write_file`, `read_file`, `replace_file`, `upload_file`
  - **Meta:** `screenshot`, `evaluate` (JavaScript execution), `done` (task completion)
- **Key features:**
  - Dynamic action registration via `@tools.action` decorator
  - Configurable action exclusion
  - Coordinate clicking toggle (auto-enabled for supported models)
  - Sensitive data detection in input actions
  - Smart file input detection (traverses DOM tree to find nearest file input)
  - JavaScript code auto-fixing (escaped quotes, XPath, selectors)
  - Direct action calling API (`tools.navigate(url=..., browser_session=...)`)

#### `browser_ai/tools/views.py` — Tool Action Models

Pydantic models for all tool action parameters.

- **Models:** `ExtractAction`, `SearchAction`, `NavigateAction`, `ClickElementAction`, `ClickElementActionIndexOnly`, `InputTextAction`, `ScrollAction`, `SendKeysAction`, `DoneAction`, `StructuredOutputAction[T]`, `SwitchTabAction`, `CloseTabAction`, `UploadFileAction`, `GetDropdownOptionsAction`, `SelectDropdownOptionAction`, `NoParamsAction`

---

### DOM

#### `browser_ai/dom/service.py` — `DomService`

Service for extracting and processing the DOM tree from browser pages using CDP snapshot, accessibility tree, and layout data.

- **Main class:** `DomService`
- **Key methods:**
  - `get_dom_tree(target_id, all_frames)` — Build enhanced DOM tree with visibility, interactivity, and accessibility data
- **Key features:**
  - Cross-origin iframe support with configurable depth limits
  - Paint order filtering for accurate element visibility
  - Accessibility tree (AX) integration for semantic element information
  - Viewport ratio detection for coordinate mapping
  - Enhanced snapshot processing (visibility, clickability, cursor styles)

#### `browser_ai/dom/views.py` — DOM Data Models

Core data structures for the DOM tree representation used throughout the system.

- **Main classes:**
  - `EnhancedDOMTreeNode` — Full DOM node with CDP node info, AX node, snapshot data, absolute position, and tree structure
  - `SimplifiedNode` — Lightweight node for serialization with interactivity flags and shadow host detection
  - `SerializedDOMState` — Complete DOM state with root node, selector map, and LLM representation
  - `DOMInteractedElement` — Element metadata for history tracking
  - `EnhancedAXNode` / `EnhancedAXProperty` — Accessibility tree node/property wrappers
  - `EnhancedSnapshotNode` — CDP snapshot data (bounds, visibility, cursor, scroll state)
- **Key features:**
  - 55+ default include attributes for element serialization
  - Node type enum (element, text, document, shadow root, etc.)
  - Selector map (index → element) for LLM element references
  - Content hashing for change detection

#### `browser_ai/dom/enhanced_snapshot.py` — Enhanced Snapshot Processing

Stateless functions for parsing Chrome DevTools Protocol DOMSnapshot data to extract visibility, clickability, cursor styles, and layout information.

- **Main function:** `build_snapshot_lookup(snapshot, device_pixel_ratio)` — Builds a lookup table of backend node ID to enhanced snapshot data
- **Key features:**
  - 10 essential computed styles tracked (display, visibility, opacity, overflow, cursor, pointer-events, position, background-color)
  - Rare boolean data parsing (input flags, text areas, etc.)
  - Bounding rect extraction with device pixel ratio normalization
  - Performance-optimized with pre-built layout index maps

#### `browser_ai/dom/markdown_extractor.py` — Markdown Extractor

Unified interface for extracting clean markdown from browser content, used by the extract action and page actor.

- **Main function:** `extract_clean_markdown(browser_session, dom_service, target_id, extract_links)`
- **Key features:**
  - HTML serialization from enhanced DOM tree
  - Markdown conversion via `markdownify` with clean output settings
  - Content noise filtering (duplicate whitespace, empty lines)
  - Content statistics tracking (HTML chars → markdown chars → filtered chars)

---

### LLM

#### `browser_ai/llm/base.py` — `BaseChatModel`

Protocol definition for all LLM chat model implementations, enabling type-safe model injection throughout the system.

- **Main class:** `BaseChatModel` (Protocol)
- **Key methods:**
  - `ainvoke(messages, output_format)` — Async invocation with optional structured output
- **Properties:** `provider`, `name`, `model_name`, `model`
- **Key features:**
  - Runtime-checkable Protocol for duck typing
  - Pydantic compatibility via custom `__get_pydantic_core_schema__`
  - Overloaded signatures for string vs structured output

#### `browser_ai/llm/models.py` — Model Registry

Convenient lazy-loaded access to all supported LLM models with auto-configuration from environment variables.

- **Main function:** `get_llm_by_name(model_name)` — Factory function to create LLM instances from string names
- **Supported providers & models:**
  - **OpenAI:** gpt-4o, gpt-4o-mini, gpt-4.1-mini, o1, o1-mini, o1-pro, o3, o3-mini, o3-pro, o4-mini, gpt-5, gpt-5-mini, gpt-5-nano
  - **Azure OpenAI:** Same model set as OpenAI
  - **Google:** gemini-2.0-flash, gemini-2.0-pro, gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
  - **Mistral:** mistral-large, mistral-medium, mistral-small, codestral, pixtral-large
  - **Cerebras:** llama3.1-8b, llama-3.3-70b, gpt-oss-120b, llama-4-scout, llama-4-maverick, qwen-3-32b, qwen-3-235b variants, qwen-3-coder-480b
  - **Browser Use:** bu-latest, bu-1.0
  - **OCI (optional):** via ChatOCIRaw
- **Key features:**
  - Lazy model instantiation via `__getattr__` (models created on first access)
  - Automatic API key injection from environment variables
  - IDE autocomplete support via type stubs

---

### Skills

#### `browser_ai/skills/service.py` — `SkillService`

Service for fetching, caching, and executing reusable automation skills from the Browser Use API.

- **Main class:** `SkillService`
- **Key methods:**
  - `async_init()` — Fetch and cache skills from API (supports wildcard `*` or specific IDs)
  - `get_skill(skill_id)` / `get_all_skills()` — Cached skill lookup
  - `execute_skill(skill_id, parameters, cookies)` — Execute with parameter validation and cookie injection
- **Key features:**
  - Pydantic parameter validation against skill schemas
  - Cookie parameter auto-injection from browser state
  - Missing cookie detection with descriptive errors
  - Pagination support for large skill libraries
  - Async SDK client management

#### `browser_ai/skills/views.py` — `Skill`

Skill data model wrapping the SDK response with helper methods for LLM integration.

- **Main class:** `Skill`
- **Key methods:**
  - `parameters_pydantic(exclude_cookies)` — Convert parameter schemas to Pydantic model
  - `output_type_pydantic` — Convert output schema to Pydantic model
  - `from_skill_response(response)` — Factory from SDK response
- **Fields:** `id`, `title`, `description`, `parameters`, `output_schema`

---

### Tokens

#### `browser_ai/tokens/service.py` — `TokenCost`

Token usage tracking and cost calculation service that fetches pricing data from LiteLLM and provides detailed per-model usage summaries.

- **Main class:** `TokenCost`
- **Key methods:**
  - `register_llm(llm)` — Wrap LLM's `ainvoke` to automatically track usage
  - `add_usage(model, usage)` — Record token usage entry
  - `calculate_cost(model, usage)` — Calculate cost breakdown (new prompt, cached, cache creation, completion)
  - `get_usage_summary(model, since)` — Comprehensive per-model statistics
  - `log_usage_summary()` — Colorized terminal output
- **Key features:**
  - Pricing data cached for 1 day with automatic refresh
  - Custom model pricing support
  - Cache-aware cost calculation (prompt caching, cache creation tokens)
  - Token formatting with k/M/B suffixes
  - Optional cost tracking (enable via `calculate_cost=True` or `BROWSER_USE_CALCULATE_COST=true`)

---

### Telemetry

#### `browser_ai/telemetry/service.py` — `ProductTelemetry`

Anonymized telemetry service using PostHog for usage analytics. Fully opt-out via `ANONYMIZED_TELEMETRY=False`.

- **Main class:** `ProductTelemetry` (singleton)
- **Key methods:**
  - `capture(event)` — Send telemetry event
  - `flush()` — Flush queued events
- **Key features:**
  - Persistent device ID stored in `~/.config/browseruse/device_id`
  - Automatic PostHog logger silencing (unless debug mode)
  - Exception auto-capture
  - GeoIP disabled for privacy

---

### Screenshots

#### `browser_ai/screenshots/service.py` — `ScreenshotService`

Simple disk-based screenshot storage service for persisting agent step screenshots.

- **Main class:** `ScreenshotService`
- **Key methods:**
  - `store_screenshot(screenshot_b64, step_number)` — Save base64 PNG to disk, return path
  - `get_screenshot(screenshot_path)` — Load from disk and return as base64
- **Storage format:** `{agent_directory}/screenshots/step_{n}.png`

---

### Filesystem

#### `browser_ai/filesystem/file_system.py` — `FileSystem`

In-memory file system with disk persistence supporting multiple file types, used by the agent to create, read, and manage files during task execution.

- **Main classes:**
  - `FileSystem` — Core file management with CRUD operations
  - `BaseFile` / `MarkdownFile` / `TxtFile` / `JsonFile` / `CsvFile` / `JsonlFile` / `PdfFile` / `DocxFile` — Type-specific file implementations
  - `FileSystemState` — Serializable state for checkpointing
- **Key methods:**
  - `write_file()` / `read_file()` / `append_file()` / `replace_file_str()` — File CRUD
  - `read_file_structured()` — Returns text + images (supports PDF, DOCX, JPG/PNG)
  - `save_extracted_content()` — Auto-numbered extracted content files
  - `describe()` — File listing with content preview (start + end for large files)
  - `get_todo_contents()` — Read the agent's todo.md file
  - `from_state()` / `get_state()` — Serialization/deserialization for checkpointing
- **Supported formats:** `.md`, `.txt`, `.json`, `.jsonl`, `.csv`, `.pdf` (via reportlab), `.docx` (via python-docx)

---

### Config

#### `browser_ai/config.py` — Configuration System

Unified configuration system with environment variables, JSON config file, and automatic migration support.

- **Main classes:**
  - `Config` (singleton as `CONFIG`) — Backward-compatible proxy that re-reads env vars on every access
  - `OldConfig` — Original lazy-loading env var configuration
  - `FlatEnvConfig` — Pydantic settings for all environment variables
  - `DBStyleConfigJSON` — Database-style JSON config with browser profiles, LLM entries, and agent entries
- **Key settings:**
  - Logging: `BROWSER_USE_LOGGING_LEVEL`, `CDP_LOGGING_LEVEL`, debug/info log files
  - Telemetry: `ANONYMIZED_TELEMETRY`, `BROWSER_USE_CLOUD_SYNC`
  - Paths: `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `BROWSER_USE_CONFIG_DIR`
  - LLM keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `AZURE_OPENAI_KEY`, etc.
  - Browser: `BROWSER_USE_PROXY_URL`, `BROWSER_USE_DISABLE_EXTENSIONS`, `BROWSER_USE_HEADLESS`
  - Runtime: `IN_DOCKER` (auto-detected), `IS_IN_EVALS`, `DEFAULT_LLM`
- **Key features:**
  - Automatic Docker detection (via /.dockerenv, PID 1 inspection, process count)
  - JSON config auto-migration from old format to database-style entries
  - Proxy configuration via environment variables

### Observability

#### `browser_ai/observability.py`

Observability decorators providing optional integration with Laminar (lmnr) for distributed tracing. Falls back to no-op wrappers when lmnr is not installed.

- **Main decorators:**
  - `observe(name, ignore_input, ignore_output, session_id)` — Traces function calls (always active when lmnr available)
  - `observe_debug(name, ...)` — Only traces when debug mode is enabled (`LMNR_LOGGING_LEVEL=debug`)
- **Key features:**
  - Zero-dependency fallback — works without lmnr installed
  - Full parameter compatibility with native lmnr `observe` decorator
  - Debug-only tracing to reduce noise in production
  - Used by 14+ core modules (agent, browser, tools, DOM, screenshots, LLM, etc.)

### Sync

#### `browser_ai/sync/auth.py`

OAuth2 Device Authorization Grant flow client for authenticating with the cloud browser service.

- **Main classes:**
  - `CloudAuthConfig` — Cloud authentication configuration (token, device_id, user_id)
  - `DeviceAuthClient` — OAuth2 device flow client (authorize, poll, refresh tokens)
- **Key features:**
  - Persistent device ID storage in `~/.config/browseruse/device_id`
  - Token refresh and expiration handling
  - Used by `browser/cloud/` for cloud browser session authentication

---

## Applications

### Task Hunter

#### `task_hunter.py`

Core task orchestration module for running browser automation tasks with profile management, task queuing, and multi-provider LLM support.

- **Key features:**
  - **Profile management:** Create, update, delete, and switch between named configurations (provider, model, proxy, headless, persistent session, etc.)
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
  - `GET /api/status/{task_id}` — Poll task status and steps
  - `POST /api/confirm/{task_id}` — Answer a pending confirmation question
  - `POST /api/profiles` — Create profile
  - `PUT /api/profiles/{name}` — Update profile
  - `GET /api/queue` — Get task queue
  - `POST /api/queue` — Add task to queue
  - `DELETE /api/queue/{task_id}` — Remove from queue
  - `POST /api/queue/run` — Run all queued tasks
- **Key features:**
  - CORS enabled for any origin
  - In-memory run status tracking with step-by-step progress
  - Confirmation flow (agent asks question → UI shows it → user responds)
  - Browser backend selection (CDP/Playwright) per run
  - Static file serving for frontend

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

### Advanced Configuration

```python
from browser_ai import Agent, Browser, BrowserProfile
from browser_ai.llm.openai.chat import ChatOpenAI

agent = Agent(
    task="Fill out the contact form on example.com",
    llm=ChatOpenAI(model="gpt-4o"),
    browser=Browser(browser_profile=BrowserProfile(
        headless=False,
        demo_mode=True,
    )),
    use_vision=True,
    max_actions_per_step=5,
    sensitive_data={"https://example.com": {"email": "user@example.com"}},
)
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
├── task_hunter.py               # Task orchestration, profiles, queue
├── static/index.html            # Web UI frontend
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
# browser_ai_web
