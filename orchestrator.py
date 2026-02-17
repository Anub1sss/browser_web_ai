"""
Task Orchestrator — средний слой между User API и Executor (Agent).

Реализует:
  - FSM (idle → planning → executing → validating → done / failed)
  - Action Graph Engine (dependency graph, ветвление, fallback, condition logic)
  - Structured Planner (граф подцелей с зависимостями и fallback-стратегиями)
  - Memory System (SQLite: хранение результатов между задачами)
  - Session Manager (Geo-proxy, GoLogin CDP adapter, session rotation)
  - Validator в цикле (Judge после каждого подшага, failure classification, feedback loop)
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


# ---------------------------------------------------------------------------
#  FSM States
# ---------------------------------------------------------------------------

class TaskState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------------------
#  Structured Planner
# ---------------------------------------------------------------------------

@dataclass
class SubGoal:
    id: int
    description: str
    status: str = "pending"  # pending | executing | done | failed | skipped
    result: str = ""
    attempts: int = 0


PLANNER_SYSTEM_PROMPT = """\
You are a planning agent. Given a user task, decompose it into a numbered list of concrete sub-goals.
Each sub-goal should be a single browser action or a small group of related actions.
Return ONLY a JSON array of strings, each string is one sub-goal. Example:
["Open DuckDuckGo and search for 'best restaurants Moscow'", "Click on the first result", "Extract the top 3 dishes from the page"]
Do NOT include any explanation outside the JSON array.
If the task is simple (1-2 actions), return 1-2 items. Maximum 10 sub-goals.
"""


async def plan_structured(llm, task: str) -> list[SubGoal]:
    """Call LLM to decompose task into structured sub-goals."""
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
        # Extract JSON array from response
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            arr = json.loads(text[start : end + 1])
            if isinstance(arr, list) and arr:
                return [SubGoal(id=i, description=str(g)) for i, g in enumerate(arr)]
        # Fallback: split by newlines
        lines = [l.strip().lstrip("0123456789.)- ") for l in text.splitlines() if l.strip()]
        if lines:
            return [SubGoal(id=i, description=l) for i, l in enumerate(lines)]
    except Exception as e:
        logger.warning("Planner error: %s", e)
    return [SubGoal(id=0, description=task)]


async def plan_simple(llm, task: str) -> list[SubGoal]:
    """Simple planner: single sub-goal = entire task."""
    return [SubGoal(id=0, description=task)]


# ---------------------------------------------------------------------------
#  Memory System (SQLite)
# ---------------------------------------------------------------------------

MEMORY_DIR = Path.home() / ".task_hunter"
MEMORY_DB = MEMORY_DIR / "memory.db"


class MemorySystem:
    """Persistent memory across tasks (SQLite)."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or MEMORY_DB)
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self.db_path)
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
            "SELECT category, key, value FROM memory WHERE task_id = ? ORDER BY id",
            (task_id,),
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


# ---------------------------------------------------------------------------
#  Session Manager (Geo + GoLogin)
# ---------------------------------------------------------------------------

# Geo-proxy presets: country code → proxy URL pattern
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
    """Resolve geo-proxy: if country is set and no custom proxy, use preset."""
    if custom_proxy and custom_proxy.get("server"):
        return custom_proxy
    if not country:
        return custom_proxy
    country = country.strip().lower()
    # Check env var first: GEO_PROXY_US, GEO_PROXY_DE, etc.
    env_key = f"GEO_PROXY_{country.upper()}"
    env_val = os.environ.get(env_key)
    if env_val:
        return {"server": env_val, "bypass": None, "username": None, "password": None}
    preset = GEO_PROXY_PRESETS.get(country)
    if preset:
        return {"server": preset, "bypass": None, "username": None, "password": None}
    return custom_proxy


async def resolve_gologin_cdp(profile_id: str, api_token: str | None = None) -> str | None:
    """
    GoLogin integration: start a GoLogin profile and return its CDP URL.
    Requires GoLogin API token (env GOLOGIN_API_TOKEN or passed directly).
    """
    token = api_token or os.environ.get("GOLOGIN_API_TOKEN")
    if not token:
        logger.warning("GoLogin API token not set (GOLOGIN_API_TOKEN)")
        return None
    if not profile_id:
        return None
    api_base = os.environ.get("GOLOGIN_API_URL", "https://api.gologin.com")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            # Start profile
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
    """Stop a running GoLogin profile."""
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


# ---------------------------------------------------------------------------
#  Validator (Judge in-loop)
# ---------------------------------------------------------------------------

VALIDATOR_SYSTEM_PROMPT = """\
You are a validation agent. Given the sub-goal description and the execution result (agent memory + actions taken),
determine if the sub-goal was achieved.
Return ONLY a JSON object: {"success": true/false, "reason": "brief explanation", "should_retry": true/false}
If the sub-goal clearly succeeded, set success=true.
If it failed but might succeed with a retry, set should_retry=true.
"""


async def validate_subgoal(llm, subgoal: SubGoal, execution_summary: str) -> dict:
    """Call LLM to validate whether a sub-goal was achieved."""
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
            return json.loads(text[start : end + 1])
    except Exception as e:
        logger.warning("Validator error: %s", e)
    return {"success": True, "reason": "Validation skipped", "should_retry": False}


# ---------------------------------------------------------------------------
#  Task Orchestrator (FSM)
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    # Planner
    enable_planner: bool = False
    # Validator (Judge in-loop)
    enable_validator: bool = False
    max_retries_per_subgoal: int = 2
    # Memory
    enable_memory: bool = True
    # Geo
    geo_country: str | None = None
    # GoLogin
    gologin_profile_id: str | None = None
    gologin_api_token: str | None = None
    # Execution
    max_steps_per_subgoal: int = 25


@dataclass
class OrchestratorContext:
    """Runtime context for a single task execution."""
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


class TaskOrchestrator:
    """
    Средний слой: FSM-оркестратор задач.

    Поток:
      idle → planning → executing → validating → (loop or done/failed)

    Опирается на существующие:
      - task_hunter.run_task (Executor = Agent + Browser + Tools)
      - browser_ai.agent.judge (Validator)
    """

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
        """
        Main entry point: run a task through the FSM.

        Args:
            task: Natural language task description
            profile_config: Browser/LLM profile config dict
            config: Orchestrator options (planner, validator, geo, gologin, memory)
            llm: LLM instance (for planner/validator; executor uses its own from profile_config)
            tools_builder: Tools factory for the executor
            step_callback: Step callback for UI updates
            task_id: Optional task ID (generated if not provided)
            state_callback: Called on every state transition
        """
        from task_hunter import get_llm, run_task as executor_run_task

        cfg = config or OrchestratorConfig()
        tid = task_id or str(uuid4())

        ctx = OrchestratorContext(
            task_id=tid,
            task=task,
            state=TaskState.IDLE,
            config=cfg,
            started_at=time.time(),
        )
        self._contexts[tid] = ctx

        def _transition(new_state: TaskState):
            ctx.state = new_state
            if state_callback:
                try:
                    state_callback(ctx)
                except Exception:
                    pass

        # Get LLM for planner/validator (reuse executor LLM if not provided)
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

        # --- GoLogin: resolve CDP URL if configured ---
        if cfg.gologin_profile_id:
            _transition(TaskState.IDLE)
            cdp_url = await resolve_gologin_cdp(cfg.gologin_profile_id, cfg.gologin_api_token)
            if cdp_url:
                ctx.gologin_cdp_url = cdp_url
                profile_config = {**profile_config, "cdp_url": cdp_url, "browser_backend": "cdp"}

        # --- Geo-proxy ---
        if cfg.geo_country:
            proxy = resolve_geo_proxy(cfg.geo_country, profile_config.get("proxy"))
            if proxy:
                profile_config = {**profile_config, "proxy": proxy}

        # --- Memory: recall relevant context ---
        memory_context = ""
        if cfg.enable_memory:
            recent = self.memory.recall(key=task[:50], limit=5)
            if recent:
                memory_context = "\n".join(
                    f"[Previous: {m['key']}] {m['value'][:200]}" for m in recent[:3]
                )

        # --- PLANNING ---
        _transition(TaskState.PLANNING)
        if cfg.enable_planner:
            ctx.plan = await plan_structured(llm, task)
        else:
            ctx.plan = await plan_simple(llm, task)

        # Log plan to memory
        if cfg.enable_memory:
            plan_text = json.dumps([g.description for g in ctx.plan], ensure_ascii=False)
            self.memory.log_task(tid, task, "planning", plan=plan_text)

        # --- EXECUTION LOOP ---
        all_succeeded = True
        for idx, subgoal in enumerate(ctx.plan):
            ctx.current_subgoal_idx = idx
            subgoal.status = "executing"
            _transition(TaskState.EXECUTING)

            # Build task text for executor
            executor_task = subgoal.description
            if memory_context:
                executor_task = f"{executor_task}\n\nContext from previous tasks:\n{memory_context}"
            if len(ctx.plan) > 1:
                progress = f"[Sub-goal {idx + 1}/{len(ctx.plan)}]"
                executor_task = f"{progress} {executor_task}"

            # Retry loop
            max_attempts = cfg.max_retries_per_subgoal + 1 if cfg.enable_validator else 1
            for attempt in range(max_attempts):
                subgoal.attempts += 1
                try:
                    ok = await executor_run_task(
                        executor_task,
                        profile_config,
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
                    execution_summary = f"Executor returned: {'success' if ok else 'failure'}. Result: {subgoal.result}"
                    verdict = await validate_subgoal(llm, subgoal, execution_summary)
                    if verdict.get("success"):
                        subgoal.status = "done"
                        break
                    elif verdict.get("should_retry") and attempt < max_attempts - 1:
                        subgoal.status = "executing"
                        logger.info("Validator: retry sub-goal %d (attempt %d)", idx, attempt + 1)
                        continue
                    else:
                        subgoal.status = "failed"
                        subgoal.result = verdict.get("reason", subgoal.result)
                        all_succeeded = False
                        break
                else:
                    subgoal.status = "done" if ok else "failed"
                    if not ok:
                        all_succeeded = False
                    break

            # Store sub-goal result in memory
            if cfg.enable_memory:
                self.memory.store(tid, subgoal.description[:100], subgoal.result, category="subgoal")

        # --- DONE / FAILED ---
        ctx.finished_at = time.time()
        if all_succeeded:
            _transition(TaskState.DONE)
        else:
            _transition(TaskState.FAILED)

        # Log final state
        if cfg.enable_memory:
            result_summary = json.dumps(
                [{"goal": g.description, "status": g.status, "result": g.result} for g in ctx.plan],
                ensure_ascii=False,
            )
            self.memory.log_task(tid, task, ctx.state.value, result=result_summary)
            self.memory.store(tid, f"task_result:{task[:80]}", ctx.state.value, category="task")

        # Stop GoLogin profile if used
        if cfg.gologin_profile_id:
            await stop_gologin_profile(cfg.gologin_profile_id, cfg.gologin_api_token)

        return ctx
