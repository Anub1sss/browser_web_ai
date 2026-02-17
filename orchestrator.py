"""
Task Orchestrator — Action Graph Engine.

Реализует:
  - FSM (idle → planning → executing → validating → done / failed)
  - Action Graph Engine:
      * dependency graph (SubGoal.depends_on)
      * ветвление и параллельные пути
      * fallback strategies (SubGoal.fallback_goal_id)
      * condition logic (on_failure, on_captcha handlers)
  - Structured Planner (граф подцелей с зависимостями)
  - Memory System (SQLite: хранение результатов между задачами)
  - Session Manager (Geo-proxy, GoLogin CDP adapter, session rotation)
  - Validator (failure classification + recommended_action)
  - Condition Handlers (captcha → rotate, blocked → proxy switch, timeout → retry)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import httpx
from dotenv import load_dotenv

load_dotenv()

from browser_ai.llm.messages import SystemMessage, UserMessage

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  FSM States
# ═══════════════════════════════════════════════════════════════════════════

class TaskState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    DONE = "done"
    FAILED = "failed"


# ═══════════════════════════════════════════════════════════════════════════
#  SubGoal — узел графа
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SubGoal:
    id: int
    description: str
    status: str = "pending"        # pending | executing | done | failed | skipped
    result: str = ""
    attempts: int = 0
    # --- Action Graph fields ---
    depends_on: list[int] = field(default_factory=list)
    fallback_goal_id: int | None = None
    on_failure: str = "fail"       # fail | skip | fallback | retry_with_strategy
    on_captcha: str = "fail"       # fail | fallback | rotate_session
    on_timeout: str = "retry"      # retry | fail | skip


# ═══════════════════════════════════════════════════════════════════════════
#  ExecutorResult — structured result from executor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutorResult:
    success: bool
    final_result: str = ""
    error_type: str | None = None  # captcha | timeout | crash | blocked | login_required
    steps_taken: int = 0


# ═══════════════════════════════════════════════════════════════════════════
#  Graph Utilities — topo sort + ready detection
# ═══════════════════════════════════════════════════════════════════════════

def topo_sort(goals: list[SubGoal]) -> list[SubGoal]:
    """Topological sort of sub-goals by depends_on. Falls back to original order on cycle."""
    id_map = {g.id: g for g in goals}
    in_degree: dict[int, int] = {g.id: 0 for g in goals}
    adj: dict[int, list[int]] = {g.id: [] for g in goals}

    for g in goals:
        for dep_id in g.depends_on:
            if dep_id in id_map:
                adj[dep_id].append(g.id)
                in_degree[g.id] += 1

    queue: deque[int] = deque(gid for gid, deg in in_degree.items() if deg == 0)
    result: list[SubGoal] = []

    while queue:
        gid = queue.popleft()
        result.append(id_map[gid])
        for nxt in adj[gid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if len(result) != len(goals):
        logger.warning("Cycle detected in sub-goal graph — falling back to original order")
        return goals
    return result


def get_ready_goals(goals: list[SubGoal]) -> list[SubGoal]:
    """Return all pending goals whose dependencies are satisfied (done/skipped)."""
    done_ids = {g.id for g in goals if g.status in ("done", "skipped")}
    ready = []
    for g in goals:
        if g.status != "pending":
            continue
        if all(dep_id in done_ids for dep_id in g.depends_on):
            ready.append(g)
    return ready


# ═══════════════════════════════════════════════════════════════════════════
#  Structured Planner — generates graph of sub-goals
# ═══════════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM_PROMPT = """\
You are a planning agent. Given a user task, decompose it into a list of sub-goals.

Return ONLY a JSON array of objects. Each object has:
- "id": integer (0-based)
- "description": string — what to do
- "depends_on": array of integer IDs this goal depends on (empty if independent)
- "on_failure": one of "fail", "skip", "fallback" (default "fail")
- "fallback_goal_id": integer ID of alternative goal if on_failure is "fallback" (null otherwise)

Example:
[
  {"id": 0, "description": "Search on DuckDuckGo for 'best restaurants'", "depends_on": [], "on_failure": "fallback", "fallback_goal_id": 1},
  {"id": 1, "description": "Search on Google for 'best restaurants' (fallback)", "depends_on": [], "on_failure": "fail", "fallback_goal_id": null},
  {"id": 2, "description": "Click first result", "depends_on": [0], "on_failure": "fail", "fallback_goal_id": null},
  {"id": 3, "description": "Extract top 3 items", "depends_on": [2], "on_failure": "fail", "fallback_goal_id": null}
]

Rules:
- Maximum 10 sub-goals.
- Simple tasks (1-2 actions): return 1-2 items.
- Use fallbacks when a search/navigation might fail.
- Do NOT include any text outside the JSON array.
"""


async def plan_structured(llm, task: str) -> list[SubGoal]:
    """Call LLM to decompose task into a graph of sub-goals."""
    try:
        system = SystemMessage(content=PLANNER_SYSTEM_PROMPT)
        user = UserMessage(content=task)
        response = await llm.ainvoke([system, user], output_format=None)
        if not response:
            return [SubGoal(id=0, description=task)]

        text = ""
        if hasattr(response, "completion") and response.completion:
            text = str(response.completion).strip()
        elif hasattr(response, "content") and response.content:
            text = str(response.content).strip()
        else:
            text = str(response).strip()

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            arr = json.loads(text[start:end + 1])
            if isinstance(arr, list) and arr:
                goals = []
                for item in arr:
                    if isinstance(item, str):
                        goals.append(SubGoal(id=len(goals), description=item))
                    elif isinstance(item, dict):
                        goals.append(SubGoal(
                            id=item.get("id", len(goals)),
                            description=item.get("description", str(item)),
                            depends_on=item.get("depends_on", []),
                            on_failure=item.get("on_failure", "fail"),
                            fallback_goal_id=item.get("fallback_goal_id"),
                            on_captcha=item.get("on_captcha", "fail"),
                            on_timeout=item.get("on_timeout", "retry"),
                        ))
                if goals:
                    return goals

        # Fallback: split by newlines
        lines = [l.strip().lstrip("0123456789.)- ") for l in text.splitlines() if l.strip()]
        if lines:
            return [SubGoal(id=i, description=l) for i, l in enumerate(lines)]
    except Exception as e:
        logger.warning("Planner error: %s", e)

    return [SubGoal(id=0, description=task)]


async def plan_simple(llm, task: str) -> list[SubGoal]:
    return [SubGoal(id=0, description=task)]


# ═══════════════════════════════════════════════════════════════════════════
#  Memory System (SQLite)
# ═══════════════════════════════════════════════════════════════════════════

MEMORY_DIR = Path.home() / ".task_hunter"
MEMORY_DB = MEMORY_DIR / "memory.db"


class MemorySystem:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or MEMORY_DB)
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'result',
                key TEXT NOT NULL,
                value TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS task_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                task TEXT NOT NULL,
                state TEXT NOT NULL,
                plan TEXT,
                result TEXT,
                started_at TEXT,
                finished_at TEXT,
                meta TEXT
            )
        """)
        self._conn.commit()

    def store(self, task_id: str, key: str, value: str, category: str = "result"):
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO memory (task_id, ts, category, key, value) VALUES (?, ?, ?, ?, ?)",
            (task_id, ts, category, key, value),
        )
        self._conn.commit()

    def recall(self, key: str | None = None, category: str | None = None, limit: int = 20) -> list[dict]:
        sql = "SELECT task_id, ts, category, key, value FROM memory WHERE 1=1"
        params: list = []
        if key:
            sql += " AND key LIKE ?"
            params.append(f"%{key}%")
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [{"task_id": r[0], "ts": r[1], "category": r[2], "key": r[3], "value": r[4]} for r in rows]

    def recall_for_task(self, task_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT category, key, value FROM memory WHERE task_id = ? ORDER BY id", (task_id,),
        ).fetchall()
        return [{"category": r[0], "key": r[1], "value": r[2]} for r in rows]

    def log_task(self, task_id: str, task: str, state: str, plan: str = "", result: str = "", meta: str = ""):
        now = datetime.now(timezone.utc).isoformat()
        existing = self._conn.execute("SELECT id FROM task_log WHERE task_id = ?", (task_id,)).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE task_log SET state=?, result=?, finished_at=?, meta=? WHERE task_id=?",
                (state, result, now, meta, task_id),
            )
        else:
            self._conn.execute(
                "INSERT INTO task_log (task_id, task, state, plan, result, started_at, meta) VALUES (?,?,?,?,?,?,?)",
                (task_id, task, state, plan, result, now, meta),
            )
        self._conn.commit()

    def get_recent_tasks(self, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT task_id, task, state, result, started_at, finished_at FROM task_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"task_id": r[0], "task": r[1], "state": r[2], "result": r[3], "started_at": r[4], "finished_at": r[5]}
            for r in rows
        ]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# ═══════════════════════════════════════════════════════════════════════════
#  Session Manager (Geo + GoLogin)
# ═══════════════════════════════════════════════════════════════════════════

GEO_PROXY_PRESETS: dict[str, str] = {
    "us": "http://us.proxy.example.com:8080",
    "de": "http://de.proxy.example.com:8080",
    "ru": "http://ru.proxy.example.com:8080",
    "uk": "http://uk.proxy.example.com:8080",
    "jp": "http://jp.proxy.example.com:8080",
    "br": "http://br.proxy.example.com:8080",
    "fr": "http://fr.proxy.example.com:8080",
    "in": "http://in.proxy.example.com:8080",
}


def resolve_geo_proxy(country: str | None, custom_proxy: dict | None = None) -> dict | None:
    if custom_proxy and custom_proxy.get("server"):
        return custom_proxy
    if not country:
        return custom_proxy
    country = country.strip().lower()
    env_key = f"GEO_PROXY_{country.upper()}"
    env_val = os.environ.get(env_key)
    if env_val:
        return {"server": env_val, "bypass": None, "username": None, "password": None}
    preset = GEO_PROXY_PRESETS.get(country)
    if preset:
        return {"server": preset, "bypass": None, "username": None, "password": None}
    return custom_proxy


async def resolve_gologin_cdp(profile_id: str, api_token: str | None = None) -> str | None:
    token = api_token or os.environ.get("GOLOGIN_API_TOKEN")
    if not token or not profile_id:
        return None
    api_base = os.environ.get("GOLOGIN_API_URL", "https://api.gologin.com")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{api_base}/browser/{profile_id}/start",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                cdp_url = data.get("wsUrl") or data.get("ws_url") or data.get("debuggerAddress")
                if cdp_url:
                    if not cdp_url.startswith("ws"):
                        cdp_url = f"ws://{cdp_url}"
                    logger.info("GoLogin profile %s started, CDP: %s", profile_id, cdp_url)
                    return cdp_url
            logger.warning("GoLogin start failed: %s %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("GoLogin error: %s", e)
    return None


async def stop_gologin_profile(profile_id: str, api_token: str | None = None):
    token = api_token or os.environ.get("GOLOGIN_API_TOKEN")
    if not token or not profile_id:
        return
    api_base = os.environ.get("GOLOGIN_API_URL", "https://api.gologin.com")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(
                f"{api_base}/browser/{profile_id}/stop",
                headers={"Authorization": f"Bearer {token}"},
            )
    except Exception as e:
        logger.debug("GoLogin stop error: %s", e)


async def rotate_browser_session(profile_config: dict, cfg) -> dict:
    """Rotate browser session: try next geo-proxy country or restart GoLogin profile."""
    countries = list(GEO_PROXY_PRESETS.keys())
    current = cfg.geo_country or ""
    if current in countries:
        idx = countries.index(current)
        next_country = countries[(idx + 1) % len(countries)]
    else:
        next_country = countries[0] if countries else None

    if next_country:
        proxy = resolve_geo_proxy(next_country, profile_config.get("proxy"))
        if proxy:
            profile_config = {**profile_config, "proxy": proxy}
            logger.info("Rotated to geo-proxy: %s", next_country)

    if cfg.gologin_profile_id:
        await stop_gologin_profile(cfg.gologin_profile_id, cfg.gologin_api_token)
        cdp_url = await resolve_gologin_cdp(cfg.gologin_profile_id, cfg.gologin_api_token)
        if cdp_url:
            profile_config = {**profile_config, "cdp_url": cdp_url}

    return profile_config


# ═══════════════════════════════════════════════════════════════════════════
#  Validator — structured failure classification
# ═══════════════════════════════════════════════════════════════════════════

VALIDATOR_SYSTEM_PROMPT = """\
You are a validation agent. Given the sub-goal description and execution result,
determine if the sub-goal was achieved and classify the failure type if it wasn't.

Return ONLY a JSON object:
{
  "success": true/false,
  "failure_type": "none" | "captcha" | "timeout" | "blocked" | "not_found" | "login_required" | "crash",
  "should_retry": true/false,
  "recommended_action": "none" | "retry" | "fallback" | "rotate_session" | "skip" | "abort",
  "reason": "brief explanation"
}
"""


async def validate_subgoal(llm, subgoal: SubGoal, execution_summary: str) -> dict:
    try:
        system = SystemMessage(content=VALIDATOR_SYSTEM_PROMPT)
        user_text = f"Sub-goal: {subgoal.description}\n\nExecution result:\n{execution_summary}"
        user = UserMessage(content=user_text)
        response = await llm.ainvoke([system, user], output_format=None)
        text = ""
        if hasattr(response, "completion") and response.completion:
            text = str(response.completion).strip()
        elif hasattr(response, "content") and response.content:
            text = str(response.content).strip()
        else:
            text = str(response).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
    except Exception as e:
        logger.warning("Validator error: %s", e)
    return {
        "success": True,
        "failure_type": "none",
        "should_retry": False,
        "recommended_action": "none",
        "reason": "Validation skipped",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Condition Handlers — dispatcher for failure types
# ═══════════════════════════════════════════════════════════════════════════

async def handle_captcha(profile_config: dict, cfg, subgoal: SubGoal) -> tuple[str, dict]:
    """IF captcha_detected → rotate session + retry."""
    logger.info("CAPTCHA detected on sub-goal %d, rotating session...", subgoal.id)
    profile_config = await rotate_browser_session(profile_config, cfg)
    return "retry", profile_config


async def handle_blocked(profile_config: dict, cfg, subgoal: SubGoal) -> tuple[str, dict]:
    """IF blocked → switch proxy or use GoLogin."""
    logger.info("Blocked detected on sub-goal %d, rotating proxy...", subgoal.id)
    profile_config = await rotate_browser_session(profile_config, cfg)
    return "retry", profile_config


async def handle_timeout(profile_config: dict, cfg, subgoal: SubGoal) -> tuple[str, dict]:
    """IF timeout → retry with same config."""
    logger.info("Timeout on sub-goal %d, retrying...", subgoal.id)
    return "retry", profile_config


async def handle_login_required(profile_config: dict, cfg, subgoal: SubGoal) -> tuple[str, dict]:
    """IF login_required → skip (can't auto-login)."""
    logger.info("Login required on sub-goal %d, skipping...", subgoal.id)
    return "skip", profile_config


CONDITION_HANDLERS = {
    "captcha": handle_captcha,
    "timeout": handle_timeout,
    "blocked": handle_blocked,
    "login_required": handle_login_required,
}


async def dispatch_failure(
    failure_type: str,
    recommended_action: str,
    subgoal: SubGoal,
    goals: list[SubGoal],
    profile_config: dict,
    cfg,
) -> tuple[str, dict]:
    """
    Dispatch failure to appropriate handler.
    Returns (action, updated_profile_config) where action is:
      "retry" | "fallback" | "skip" | "fail"
    """
    # 1. Try condition handler for the failure type
    handler = CONDITION_HANDLERS.get(failure_type)
    if handler:
        action, profile_config = await handler(profile_config, cfg, subgoal)
        if action != "fail":
            return action, profile_config

    # 2. Use recommended_action from validator
    if recommended_action == "rotate_session":
        profile_config = await rotate_browser_session(profile_config, cfg)
        return "retry", profile_config
    if recommended_action == "fallback" and subgoal.fallback_goal_id is not None:
        return "fallback", profile_config
    if recommended_action == "skip":
        return "skip", profile_config
    if recommended_action == "retry":
        return "retry", profile_config

    # 3. Use sub-goal's own on_failure strategy
    if subgoal.on_failure == "skip":
        return "skip", profile_config
    if subgoal.on_failure == "fallback" and subgoal.fallback_goal_id is not None:
        return "fallback", profile_config

    return "fail", profile_config


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator Config + Context
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OrchestratorConfig:
    enable_planner: bool = False
    enable_validator: bool = False
    max_retries_per_subgoal: int = 2
    enable_memory: bool = True
    geo_country: str | None = None
    gologin_profile_id: str | None = None
    gologin_api_token: str | None = None
    max_steps_per_subgoal: int = 25


@dataclass
class OrchestratorContext:
    task_id: str = ""
    task: str = ""
    state: TaskState = TaskState.IDLE
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    plan: list[SubGoal] = field(default_factory=list)
    current_subgoal_idx: int = 0
    memory_entries: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    gologin_cdp_url: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
#  Task Orchestrator — Graph Execution Engine
# ═══════════════════════════════════════════════════════════════════════════

class TaskOrchestrator:
    def __init__(self, memory: MemorySystem | None = None):
        self.memory = memory or MemorySystem()
        self._contexts: dict[str, OrchestratorContext] = {}

    def get_context(self, task_id: str) -> OrchestratorContext | None:
        return self._contexts.get(task_id)

    async def run(
        self,
        task: str,
        profile_config: dict,
        config: OrchestratorConfig | None = None,
        llm=None,
        tools_builder=None,
        step_callback=None,
        task_id: str | None = None,
        state_callback: Callable[[OrchestratorContext], None] | None = None,
    ) -> OrchestratorContext:
        from task_hunter import get_llm, run_task as executor_run_task

        cfg = config or OrchestratorConfig()
        tid = task_id or str(uuid4())

        ctx = OrchestratorContext(
            task_id=tid, task=task, state=TaskState.IDLE,
            config=cfg, started_at=time.time(),
        )
        self._contexts[tid] = ctx

        def _transition(new_state: TaskState):
            ctx.state = new_state
            if state_callback:
                try:
                    state_callback(ctx)
                except Exception:
                    pass

        # --- Get LLM ---
        if llm is None:
            try:
                llm = get_llm(
                    profile_config.get("provider", "custom"),
                    model=profile_config.get("model"),
                    temperature=profile_config.get("temperature"),
                    max_completion_tokens=profile_config.get("max_completion_tokens"),
                )
            except Exception as e:
                ctx.error = f"LLM init failed: {e}"
                _transition(TaskState.FAILED)
                return ctx

        # --- GoLogin ---
        if cfg.gologin_profile_id:
            cdp_url = await resolve_gologin_cdp(cfg.gologin_profile_id, cfg.gologin_api_token)
            if cdp_url:
                ctx.gologin_cdp_url = cdp_url
                profile_config = {**profile_config, "cdp_url": cdp_url, "browser_backend": "cdp"}

        # --- Geo-proxy ---
        if cfg.geo_country:
            proxy = resolve_geo_proxy(cfg.geo_country, profile_config.get("proxy"))
            if proxy:
                profile_config = {**profile_config, "proxy": proxy}

        # --- Memory context ---
        memory_context = ""
        if cfg.enable_memory:
            recent = self.memory.recall(key=task[:50], limit=5)
            if recent:
                memory_context = "\n".join(
                    f"[Previous: {m['key']}] {m['value'][:200]}" for m in recent[:3]
                )

        # ═══════════════════════════════════════════════════════════════
        #  PLANNING PHASE
        # ═══════════════════════════════════════════════════════════════
        _transition(TaskState.PLANNING)
        if cfg.enable_planner:
            ctx.plan = await plan_structured(llm, task)
        else:
            ctx.plan = await plan_simple(llm, task)

        if cfg.enable_memory:
            plan_text = json.dumps([g.description for g in ctx.plan], ensure_ascii=False)
            self.memory.log_task(tid, task, "planning", plan=plan_text)

        # Topological sort
        sorted_goals = topo_sort(ctx.plan)
        ctx.plan = sorted_goals

        # ═══════════════════════════════════════════════════════════════
        #  GRAPH EXECUTION LOOP
        # ═══════════════════════════════════════════════════════════════
        all_succeeded = True
        max_iterations = len(ctx.plan) * (cfg.max_retries_per_subgoal + 2) + 5  # safety limit

        for _ in range(max_iterations):
            ready = get_ready_goals(ctx.plan)
            if not ready:
                # Check if all goals are done/skipped or if we're stuck
                pending = [g for g in ctx.plan if g.status == "pending"]
                if not pending:
                    break  # All goals processed
                # Stuck — dependencies can't be resolved
                logger.warning("Graph stuck: %d pending goals with unresolved deps", len(pending))
                for g in pending:
                    g.status = "failed"
                    g.result = "Unresolved dependencies"
                all_succeeded = False
                break

            # Execute first ready goal (sequential for now; parallel possible later)
            subgoal = ready[0]
            ctx.current_subgoal_idx = subgoal.id
            subgoal.status = "executing"
            _transition(TaskState.EXECUTING)

            # Build task text
            executor_task = subgoal.description
            if memory_context:
                executor_task = f"{executor_task}\n\nContext from previous tasks:\n{memory_context}"
            if len(ctx.plan) > 1:
                executor_task = f"[Sub-goal {subgoal.id + 1}/{len(ctx.plan)}] {executor_task}"

            # --- Execute with retry ---
            max_attempts = cfg.max_retries_per_subgoal + 1 if cfg.enable_validator else 1
            attempt = 0
            goal_resolved = False

            while attempt < max_attempts and not goal_resolved:
                subgoal.attempts += 1
                attempt += 1

                try:
                    ok = await executor_run_task(
                        executor_task, profile_config,
                        tools_builder=tools_builder,
                        step_callback=step_callback,
                        task_id=tid,
                    )
                    subgoal.result = "success" if ok else "failed"
                except Exception as e:
                    ok = False
                    subgoal.result = f"error: {e}"

                # --- VALIDATION ---
                if cfg.enable_validator and llm:
                    _transition(TaskState.VALIDATING)
                    exec_summary = f"Executor: {'success' if ok else 'failure'}. Result: {subgoal.result}"
                    verdict = await validate_subgoal(llm, subgoal, exec_summary)

                    if verdict.get("success"):
                        subgoal.status = "done"
                        goal_resolved = True
                    else:
                        failure_type = verdict.get("failure_type", "none")
                        recommended = verdict.get("recommended_action", "fail")

                        action, profile_config = await dispatch_failure(
                            failure_type, recommended, subgoal, ctx.plan, profile_config, cfg,
                        )

                        if action == "retry" and attempt < max_attempts:
                            subgoal.status = "executing"
                            logger.info("Retrying sub-goal %d (attempt %d/%d)", subgoal.id, attempt + 1, max_attempts)
                            continue
                        elif action == "fallback" and subgoal.fallback_goal_id is not None:
                            subgoal.status = "failed"
                            subgoal.result = verdict.get("reason", subgoal.result)
                            # Activate fallback goal
                            fb = next((g for g in ctx.plan if g.id == subgoal.fallback_goal_id), None)
                            if fb and fb.status == "pending":
                                logger.info("Activating fallback goal %d for failed goal %d", fb.id, subgoal.id)
                            goal_resolved = True
                        elif action == "skip":
                            subgoal.status = "skipped"
                            goal_resolved = True
                        else:
                            subgoal.status = "failed"
                            subgoal.result = verdict.get("reason", subgoal.result)
                            all_succeeded = False
                            goal_resolved = True
                else:
                    # No validator
                    subgoal.status = "done" if ok else "failed"
                    if not ok:
                        # Apply on_failure strategy without validator
                        if subgoal.on_failure == "skip":
                            subgoal.status = "skipped"
                        elif subgoal.on_failure == "fallback" and subgoal.fallback_goal_id is not None:
                            subgoal.status = "failed"
                        else:
                            all_succeeded = False
                    goal_resolved = True

            # Store result in memory
            if cfg.enable_memory:
                self.memory.store(tid, subgoal.description[:100], subgoal.result, category="subgoal")

        # ═══════════════════════════════════════════════════════════════
        #  FINALIZE
        # ═══════════════════════════════════════════════════════════════
        ctx.finished_at = time.time()

        failed_goals = [g for g in ctx.plan if g.status == "failed"]
        if failed_goals:
            all_succeeded = False

        _transition(TaskState.DONE if all_succeeded else TaskState.FAILED)

        if cfg.enable_memory:
            result_summary = json.dumps(
                [{"goal": g.description, "status": g.status, "result": g.result} for g in ctx.plan],
                ensure_ascii=False,
            )
            self.memory.log_task(tid, task, ctx.state.value, result=result_summary)
            self.memory.store(tid, f"task_result:{task[:80]}", ctx.state.value, category="task")

        if cfg.gologin_profile_id:
            await stop_gologin_profile(cfg.gologin_profile_id, cfg.gologin_api_token)

        return ctx
