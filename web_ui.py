"""
Task Hunter — веб-интерфейс.
Запуск: python web_ui.py  или  uvicorn web_ui:app --reload --host 0.0.0.0 --port 8765
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import after env is loaded so browser_ai sees correct env
load_dotenv()

from browser_ai.agent.views import ActionResult
from browser_ai.tools.service import Tools

from orchestrator import (
	GEO_PROXY_PRESETS,
	MemorySystem,
	OrchestratorConfig,
	TaskOrchestrator,
	TaskState,
)
from task_hunter import (
	DEFAULT_PROFILE_NAME,
	configure_logs,
	get_profile_config,
	load_profiles,
	load_queue,
	run_task,
	save_profiles,
	save_queue,
)

# Shared orchestrator and memory instances
_memory = MemorySystem()
_orchestrator = TaskOrchestrator(memory=_memory)

# In-memory run status: task_id -> { status, message, steps[], pending_confirm?, _event, _confirm_result }
run_status: dict[str, dict] = {}


def _make_step_callback(task_id: str):
	"""Build step callback that appends to run_status[task_id]['steps']."""
	async def _step_callback(state_summary, agent_output, step_number: int) -> None:
		if task_id not in run_status:
			return
		state = agent_output.current_state
		actions = []
		for action in agent_output.action:
			action_data = action.model_dump(exclude_unset=True)
			action_name = next(iter(action_data.keys())) if action_data else "action"
			params = action_data.get(action_name, {})
			actions.append({"name": action_name, "params": params})
		url = getattr(state_summary, "url", None) or getattr(state_summary, "current_url", None)
		run_status[task_id].setdefault("steps", []).append({
			"step_number": step_number,
			"memory": getattr(state, "memory", None) or "",
			"next_goal": getattr(state, "next_goal", None) or "",
			"actions": actions,
			"url": url,
		})

	return _step_callback


def build_tools_web_with_confirm(task_id: str) -> Tools:
	"""Tools for web UI: confirm_destructive shows question on site and waits for user answer."""
	tools = Tools()
	status = run_status.get(task_id)
	if not status:
		# Fallback: auto-approve
		@tools.action(description="Ask user to complete a manual step and confirm.")
		async def wait_for_user(step: str) -> ActionResult:
			return ActionResult(extracted_content="ok", long_term_memory="User confirmed (web).")

		@tools.action(description="Ask user to confirm a destructive action before proceeding.")
		async def confirm_destructive(question: str) -> ActionResult:
			return ActionResult(extracted_content="yes")
		return tools

	@tools.action(description="Ask user to complete a manual step and confirm.")
	async def wait_for_user(step: str) -> ActionResult:
		return ActionResult(extracted_content="ok", long_term_memory="User confirmed (web).")

	@tools.action(description="Ask user to confirm a destructive action before proceeding.")
	async def confirm_destructive(question: str) -> ActionResult:
		status["pending_confirm"] = {"question": question}
		ev = status.get("_event")
		if ev:
			await ev.wait()
		result = (status.get("_confirm_result") or "yes").strip().lower()
		status["pending_confirm"] = None
		if result in ("yes", "y", "да", "ok"):
			return ActionResult(extracted_content="yes")
		return ActionResult(extracted_content="no")

	return tools


def _ensure_static_dir() -> Path:
	d = Path(__file__).resolve().parent / "static"
	d.mkdir(exist_ok=True)
	return d


@asynccontextmanager
async def lifespan(app: FastAPI):
	configure_logs()
	yield
	# cleanup if needed


app = FastAPI(title="Task Hunter", lifespan=lifespan)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


# --- API models ---
class RunTaskRequest(BaseModel):
	task: str
	profile_name: str = DEFAULT_PROFILE_NAME
	provider: str | None = None
	model: str | None = None
	browser_backend: str | None = None  # "cdp" | "playwright"
	playwright_browser: str | None = None  # "chromium" | "firefox" | "webkit"
	# Orchestrator options
	enable_planner: bool = False
	enable_validator: bool = False
	enable_memory: bool = True
	geo_country: str | None = None  # "us", "de", "ru", etc.
	gologin_profile_id: str | None = None
	gologin_api_token: str | None = None
	max_steps_per_subgoal: int = 25
	max_retries: int = 2


class ConfirmRequest(BaseModel):
	answer: str  # "yes" | "no"


class ProfileCreate(BaseModel):
	name: str
	provider: str = "custom"
	model: str | None = None
	headless: bool = False
	persistent_session: bool = False


# --- API routes ---
@app.get("/api/profiles")
def api_profiles():
	data = load_profiles()
	return {
		"default_profile": data.get("default_profile", DEFAULT_PROFILE_NAME),
		"profiles": list(data.get("profiles", {}).keys()),
		"profiles_detail": data.get("profiles", {}),
	}


@app.get("/api/profiles/{name}")
def api_profile_get(name: str):
	data = load_profiles()
	if name not in data.get("profiles", {}):
		raise HTTPException(status_code=404, detail="Profile not found")
	return get_profile_config(data, name)


@app.post("/api/run")
async def api_run(req: RunTaskRequest):
	if not (req.task or req.task.strip()):
		raise HTTPException(status_code=400, detail="Task is required")
	profiles = load_profiles()
	profile_config = get_profile_config(profiles, req.profile_name or DEFAULT_PROFILE_NAME)
	if req.provider is not None:
		profile_config = {**profile_config, "provider": req.provider}
	if req.model is not None:
		profile_config = {**profile_config, "model": req.model}
	# Browser backend
	backend = (req.browser_backend or "").strip().lower()
	if backend in ("cdp", "playwright"):
		profile_config = {**profile_config, "browser_backend": backend}
		if backend == "playwright":
			if req.playwright_browser and req.playwright_browser.strip():
				pb = req.playwright_browser.strip().lower()
			else:
				pb = profile_config.get("playwright_browser", "chromium")
			if pb not in ("chromium", "firefox", "webkit"):
				pb = "chromium"
			profile_config = {**profile_config, "playwright_browser": pb}

	task_id = str(uuid4())
	ev = asyncio.Event()
	# Build orchestrator config from request
	orch_config = OrchestratorConfig(
		enable_planner=req.enable_planner,
		enable_validator=req.enable_validator,
		max_retries_per_subgoal=req.max_retries,
		enable_memory=req.enable_memory,
		geo_country=req.geo_country,
		gologin_profile_id=req.gologin_profile_id,
		gologin_api_token=req.gologin_api_token,
		max_steps_per_subgoal=req.max_steps_per_subgoal,
	)
	use_orchestrator = req.enable_planner or req.enable_validator or req.geo_country or req.gologin_profile_id

	run_status[task_id] = {
		"status": "running",
		"message": "Starting...",
		"steps": [],
		"pending_confirm": None,
		"_event": ev,
		"_confirm_result": None,
		"orchestrator": {
			"state": "idle",
			"plan": [],
			"current_subgoal": 0,
			"enabled": use_orchestrator,
		},
	}
	step_callback = _make_step_callback(task_id)

	def _state_callback(ctx):
		"""Update orchestrator state in run_status for UI polling."""
		if task_id in run_status:
			run_status[task_id]["orchestrator"] = {
				"state": ctx.state.value,
				"plan": [
					{"id": g.id, "description": g.description, "status": g.status, "result": g.result}
					for g in ctx.plan
				],
				"current_subgoal": ctx.current_subgoal_idx,
				"enabled": True,
			}

	async def _run():
		try:
			if use_orchestrator:
				ctx = await _orchestrator.run(
					task=req.task.strip(),
					profile_config=profile_config,
					config=orch_config,
					tools_builder=build_tools_web_with_confirm,
					step_callback=step_callback,
					task_id=task_id,
					state_callback=_state_callback,
				)
				ok = ctx.state == TaskState.DONE
				run_status[task_id]["status"] = "done" if ok else "error"
				run_status[task_id]["message"] = (
					"Task completed successfully." if ok else f"Task failed: {ctx.error or 'sub-goal failed'}"
				)
			else:
				ok = await run_task(
					req.task.strip(),
					profile_config,
					tools_builder=build_tools_web_with_confirm,
					step_callback=step_callback,
					task_id=task_id,
				)
				run_status[task_id]["status"] = "done" if ok else "error"
				run_status[task_id]["message"] = "Task completed successfully." if ok else "Task failed."
		except Exception as e:
			logging.exception("Run task failed")
			run_status[task_id]["status"] = "error"
			run_status[task_id]["message"] = str(e)
		ev.set()

	asyncio.create_task(_run())
	return {"task_id": task_id, "status": "running"}


def _safe_run_status(task_id: str) -> dict:
	"""Return run status without internal fields for API response."""
	if task_id not in run_status:
		raise HTTPException(status_code=404, detail="Task not found")
	data = run_status[task_id].copy()
	data.pop("_event", None)
	data.pop("_confirm_result", None)
	return data


@app.get("/api/run/{task_id}")
def api_run_status(task_id: str):
	return _safe_run_status(task_id)


@app.post("/api/run/{task_id}/confirm")
async def api_run_confirm(task_id: str, body: ConfirmRequest):
	if task_id not in run_status:
		raise HTTPException(status_code=404, detail="Task not found")
	data = run_status[task_id]
	if not data.get("pending_confirm"):
		raise HTTPException(status_code=400, detail="No pending confirmation")
	data["_confirm_result"] = body.answer.strip()
	data["_event"].set()
	return {"ok": True}


@app.get("/api/queue")
def api_queue():
	return load_queue()


@app.post("/api/queue")
def api_queue_add(item: dict):
	queue = load_queue()
	queue.setdefault("items", [])
	item.setdefault("id", str(uuid4()))
	item.setdefault("status", "pending")
	item.setdefault("priority", 0)
	item.setdefault("retries_left", 0)
	item.setdefault("attempts", 0)
	item.setdefault("profile", DEFAULT_PROFILE_NAME)
	queue["items"].append(item)
	save_queue(queue)
	return {"ok": True, "queue": queue}


# --- Memory API ---
@app.get("/api/memory/recent")
def api_memory_recent(limit: int = 20):
	return {"entries": _memory.recall(limit=limit)}


@app.get("/api/memory/tasks")
def api_memory_tasks(limit: int = 20):
	return {"tasks": _memory.get_recent_tasks(limit=limit)}


@app.get("/api/memory/search")
def api_memory_search(q: str = "", category: str = ""):
	return {"entries": _memory.recall(key=q or None, category=category or None, limit=50)}


# --- Geo presets ---
@app.get("/api/geo/countries")
def api_geo_countries():
	return {"countries": sorted(GEO_PROXY_PRESETS.keys())}


# --- Serve frontend ---
static_dir = _ensure_static_dir()
index_path = static_dir / "index.html"


@app.get("/")
def index():
	if index_path.exists():
		return FileResponse(index_path)
	raise HTTPException(status_code=404, detail="static/index.html not found")


if __name__ == "__main__":
	import uvicorn
	# Ensure static/index.html exists before opening browser
	if not index_path.exists():
		logging.warning("static/index.html missing — create frontend first")
	uvicorn.run(app, host="0.0.0.0", port=8765)
