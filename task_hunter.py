import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

# Optional: log every file open (read/write) when BROWSER_USE_LOG_FILE_ACCESS=1
if os.environ.get("BROWSER_USE_LOG_FILE_ACCESS", "").strip().lower() in ("1", "true", "yes"):
	import builtins
	_original_open = builtins.open
	def _logged_open(file, mode="r", *args, **kwargs):
		path = getattr(file, "name", file) if hasattr(file, "name") else file
		if isinstance(path, (str, Path)):
			print(f"[file] {mode}: {path}", flush=True)
		return _original_open(file, mode, *args, **kwargs)
	builtins.open = _logged_open

from browser_ai import Agent, Browser, ChatBrowserUse, ChatOpenAI
from browser_ai.agent.views import ActionResult
from browser_ai.browser.profile import BrowserProfile, ProxySettings
from browser_ai.llm.glm.chat import ChatGLM
from browser_ai.llm.messages import SystemMessage, UserMessage
from browser_ai.tools.service import Tools

DEFAULT_PROFILE_DIR = ".browser_profile"
SCREENSHOT_DIR = "artifacts/screenshots"
STUCK_REPEAT_THRESHOLD = 3
TASK_HUNTER_DIR = Path.home() / ".task_hunter"
PROFILES_FILE = TASK_HUNTER_DIR / "profiles.json"
QUEUE_FILE = TASK_HUNTER_DIR / "queue.json"
DEFAULT_START_DELAY_SECONDS = 8
DEFAULT_ACTION_DELAY_SECONDS = 2.0
DEFAULT_PROFILE_NAME = "default"

DEFAULT_PROFILE_CONFIG = {
	"provider": "custom",
	"model": None,
	"temperature": None,
	"max_completion_tokens": None,
	"persistent_session": False,
	"profile_dir": DEFAULT_PROFILE_DIR,
	"headless": False,
	"wait_between_actions": DEFAULT_ACTION_DELAY_SECONDS,
	"start_delay_seconds": DEFAULT_START_DELAY_SECONDS,
	"proxy": {
		"server": None,
		"bypass": None,
		"username": None,
		"password": None,
	},
	"enable_fallback": False,
	"enable_subagents": False,
	"browser_backend": "cdp",
	"playwright_browser": "chromium",
}


def _ensure_data_dir() -> None:
	TASK_HUNTER_DIR.mkdir(parents=True, exist_ok=True)


def load_profiles() -> dict:
	_ensure_data_dir()
	if PROFILES_FILE.exists():
		try:
			return json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
		except Exception:
			pass
	default_profile = DEFAULT_PROFILE_CONFIG.copy()
	default_profile["proxy"] = DEFAULT_PROFILE_CONFIG["proxy"].copy()
	data = {"default_profile": DEFAULT_PROFILE_NAME, "profiles": {DEFAULT_PROFILE_NAME: default_profile}}
	save_profiles(data)
	return data


def save_profiles(data: dict) -> None:
	_ensure_data_dir()
	PROFILES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_queue() -> dict:
	_ensure_data_dir()
	if QUEUE_FILE.exists():
		try:
			return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
		except Exception:
			pass
	data = {"paused": False, "items": []}
	save_queue(data)
	return data


def save_queue(data: dict) -> None:
	_ensure_data_dir()
	QUEUE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _prompt_str(prompt: str, default: str | None = None) -> str:
	if default is None:
		return input(prompt).strip()
	value = input(f"{prompt} [{default}]: ").strip()
	return value or default


def _prompt_bool(prompt: str, default: bool = False) -> bool:
	default_str = "y" if default else "n"
	value = input(f"{prompt} (y/n) [{default_str}]: ").strip().lower()
	if not value:
		return default
	return value in {"y", "yes", "да", "1", "true"}


def _prompt_int(prompt: str, default: int | None = None) -> int | None:
	default_str = "" if default is None else str(default)
	raw = input(f"{prompt}{' [' + default_str + ']' if default is not None else ''}: ").strip()
	if not raw:
		return default
	try:
		return int(raw)
	except ValueError:
		return default


def _prompt_float(prompt: str, default: float | None = None) -> float | None:
	default_str = "" if default is None else str(default)
	raw = input(f"{prompt}{' [' + default_str + ']' if default is not None else ''}: ").strip()
	if not raw:
		return default
	try:
		return float(raw)
	except ValueError:
		return default


def select_profile(profiles: dict, prompt: str = "Select profile") -> str:
	names = sorted(profiles["profiles"].keys())
	default_name = profiles.get("default_profile", DEFAULT_PROFILE_NAME)
	print("Available profiles:", ", ".join(names), flush=True)
	choice = _prompt_str(f"{prompt}", default_name)
	if choice not in profiles["profiles"]:
		print(f"Profile '{choice}' not found, using default '{default_name}'", flush=True)
		return default_name
	return choice


def get_profile_config(profiles: dict, name: str) -> dict:
	profile = profiles["profiles"].get(name) or profiles["profiles"].get(profiles.get("default_profile"))
	if not profile:
		fallback = DEFAULT_PROFILE_CONFIG.copy()
		fallback["proxy"] = DEFAULT_PROFILE_CONFIG["proxy"].copy()
		return fallback
	merged = DEFAULT_PROFILE_CONFIG.copy()
	merged["proxy"] = DEFAULT_PROFILE_CONFIG["proxy"].copy()
	merged.update(profile)
	if "proxy" in profile and isinstance(profile["proxy"], dict):
		proxy = DEFAULT_PROFILE_CONFIG["proxy"].copy()
		proxy.update(profile["proxy"])
		merged["proxy"] = proxy
	return merged


def list_profiles(profiles: dict) -> None:
	default_name = profiles.get("default_profile", DEFAULT_PROFILE_NAME)
	for name, profile in profiles["profiles"].items():
		tag = " (default)" if name == default_name else ""
		print(f"- {name}{tag}: provider={profile.get('provider')}, model={profile.get('model')}", flush=True)


def upsert_profile(profiles: dict) -> None:
	name = _prompt_str("Profile name").strip()
	if not name:
		print("Profile name is required", flush=True)
		return
	if name in profiles["profiles"]:
		current = get_profile_config(profiles, name)
	else:
		current = DEFAULT_PROFILE_CONFIG.copy()
		current["proxy"] = DEFAULT_PROFILE_CONFIG["proxy"].copy()
	provider = _prompt_str("Provider (custom/openai/glm)", current.get("provider", "custom"))
	model = _prompt_str("Model (optional)", current.get("model") or "")
	temperature = _prompt_float("Temperature (optional)", current.get("temperature"))
	max_tokens = _prompt_int("Max completion tokens (optional)", current.get("max_completion_tokens"))
	headless = _prompt_bool("Headless browser", current.get("headless", False))
	persistent_session = _prompt_bool("Persistent session", current.get("persistent_session", False))
	profile_dir = _prompt_str("Profile directory", current.get("profile_dir", DEFAULT_PROFILE_DIR))
	wait_between_actions = _prompt_float("Wait between actions (seconds)", current.get("wait_between_actions"))
	start_delay_seconds = _prompt_float("Start delay before actions (seconds)", current.get("start_delay_seconds"))
	enable_fallback = _prompt_bool("Enable fallback LLM", current.get("enable_fallback", False))
	enable_subagents = _prompt_bool("Enable sub-agent planning", current.get("enable_subagents", False))

	proxy_server = _prompt_str("Proxy server (optional)", current.get("proxy", {}).get("server") or "")
	proxy_bypass = _prompt_str("Proxy bypass (optional)", current.get("proxy", {}).get("bypass") or "")
	proxy_username = _prompt_str("Proxy username (optional)", current.get("proxy", {}).get("username") or "")
	proxy_password = _prompt_str("Proxy password (optional)", current.get("proxy", {}).get("password") or "")

	profiles["profiles"][name] = {
		"provider": provider,
		"model": model or None,
		"temperature": temperature,
		"max_completion_tokens": max_tokens,
		"persistent_session": persistent_session,
		"profile_dir": profile_dir or DEFAULT_PROFILE_DIR,
		"headless": headless,
		"wait_between_actions": wait_between_actions
		if wait_between_actions is not None
		else DEFAULT_ACTION_DELAY_SECONDS,
		"start_delay_seconds": start_delay_seconds
		if start_delay_seconds is not None
		else DEFAULT_START_DELAY_SECONDS,
		"proxy": {
			"server": proxy_server or None,
			"bypass": proxy_bypass or None,
			"username": proxy_username or None,
			"password": proxy_password or None,
		},
		"enable_fallback": enable_fallback,
		"enable_subagents": enable_subagents,
	}
	save_profiles(profiles)
	print(f"Saved profile '{name}'", flush=True)


def set_default_profile(profiles: dict) -> None:
	name = select_profile(profiles, "Set default profile")
	profiles["default_profile"] = name
	save_profiles(profiles)
	print(f"Default profile set to '{name}'", flush=True)


def delete_profile(profiles: dict) -> None:
	name = select_profile(profiles, "Delete profile")
	if name == profiles.get("default_profile") and len(profiles["profiles"]) > 1:
		print("Cannot delete default profile. Set another default first.", flush=True)
		return
	profiles["profiles"].pop(name, None)
	if not profiles["profiles"]:
		default_profile = DEFAULT_PROFILE_CONFIG.copy()
		default_profile["proxy"] = DEFAULT_PROFILE_CONFIG["proxy"].copy()
		profiles["profiles"][DEFAULT_PROFILE_NAME] = default_profile
		profiles["default_profile"] = DEFAULT_PROFILE_NAME
	save_profiles(profiles)
	print(f"Deleted profile '{name}'", flush=True)


def profile_menu() -> None:
	profiles = load_profiles()
	while True:
		print("\nProfile menu:", flush=True)
		print("1) List profiles", flush=True)
		print("2) Create/Update profile", flush=True)
		print("3) Set default profile", flush=True)
		print("4) Delete profile", flush=True)
		print("5) Back", flush=True)
		choice = input("Select option: ").strip()
		if choice == "1":
			list_profiles(profiles)
		elif choice == "2":
			upsert_profile(profiles)
		elif choice == "3":
			set_default_profile(profiles)
		elif choice == "4":
			delete_profile(profiles)
		elif choice == "5":
			return
		else:
			print("Unknown option", flush=True)


def configure_logs() -> None:
	os.environ['BROWSER_USE_SILENT_BACKGROUND_EXCEPTIONS'] = '1'
	logging.getLogger().setLevel(logging.CRITICAL)
	logging.getLogger('browser_use').setLevel(logging.CRITICAL)
	logging.getLogger('browser_ai.llm').setLevel(logging.CRITICAL)
	logging.getLogger('cdp_use').setLevel(logging.CRITICAL)
	logging.getLogger('asyncio').setLevel(logging.CRITICAL)
	logging.getLogger('bubus').setLevel(logging.CRITICAL)


def get_llm(
	provider: str,
	model: str | None = None,
	temperature: float | None = None,
	max_completion_tokens: int | None = None,
):
	"""Get LLM instance based on provider choice."""
	provider = provider.lower().strip()

	if provider in {'custom', 'custom-llm', 'cllm'}:
		return ChatBrowserUse(model=model or 'bu-latest')
	if provider == 'openai':
		api_key = os.getenv('OPENAI_API_KEY')
		if not api_key:
			raise ValueError('OPENAI_API_KEY environment variable is required for OpenAI provider')
		return ChatOpenAI(
			model=model or 'gpt-4o',
			api_key=api_key,
			temperature=temperature,
			max_completion_tokens=max_completion_tokens,
		)
	if provider == 'glm':
		api_key = os.getenv('ZAI_API_KEY')
		if not api_key:
			raise ValueError('ZAI_API_KEY environment variable is required for GLM provider')
		return ChatGLM(model=model or 'GLM-4.6V-Flash', api_key=api_key, temperature=temperature)
	raise ValueError(f'Unknown provider: {provider}. Available: custom, openai, glm')


def build_tools() -> Tools[None]:
	tools = Tools()

	@tools.action(description='Ask user to complete a manual step and confirm.')
	async def wait_for_user(step: str) -> ActionResult:
		print(step, flush=True)
		answer = input('(Press Enter when done) > ')
		return ActionResult(extracted_content=answer or 'ok', long_term_memory='User confirmed manual step.')

	@tools.action(description='Ask user to confirm a destructive action before proceeding.')
	async def confirm_destructive(question: str) -> ActionResult:
		print(question, flush=True)
		answer = input('Type "yes" or "no" > ')
		normalized = answer.strip().lower()
		if normalized in {'yes', 'y', 'да', 'ok', 'ага', 'конечно'}:
			return ActionResult(extracted_content='yes')
		return ActionResult(extracted_content='no')

	return tools


def build_tools_web() -> Tools[None]:
	"""Tools for web UI: no stdin; wait_for_user and confirm_destructive auto-approve."""
	tools = Tools()

	@tools.action(description='Ask user to complete a manual step and confirm.')
	async def wait_for_user(step: str) -> ActionResult:
		print(step, flush=True)
		return ActionResult(extracted_content='ok', long_term_memory='User confirmed manual step (web).')

	@tools.action(description='Ask user to confirm a destructive action before proceeding.')
	async def confirm_destructive(question: str) -> ActionResult:
		print(question, flush=True)
		return ActionResult(extracted_content='yes')

	return tools


def build_browser(
	persistent: bool,
	profile_dir: str | None,
	headless: bool,
	wait_between_actions: float | None,
	proxy: dict | None,
	browser_backend: str = "cdp",
	playwright_browser: str = "chromium",
) -> Browser:
	"""Create a visible browser session, optionally persistent. Use browser_backend='playwright' for Firefox/WebKit."""
	user_data_dir = None
	if persistent and browser_backend == "cdp":
		dir_path = Path(profile_dir or DEFAULT_PROFILE_DIR).expanduser()
		dir_path.mkdir(parents=True, exist_ok=True)
		user_data_dir = str(dir_path)
	proxy_settings = None
	if proxy and any(proxy.values()):
		proxy_settings = ProxySettings(
			server=proxy.get("server"),
			bypass=proxy.get("bypass"),
			username=proxy.get("username"),
			password=proxy.get("password"),
		)
	profile = BrowserProfile(
		headless=headless,
		user_data_dir=user_data_dir,
		wait_between_actions=wait_between_actions,
		proxy=proxy_settings,
		browser_backend=browser_backend,
		playwright_browser=playwright_browser,
	)
	driver = None
	if browser_backend == "playwright":
		from browser_ai.browser.driver import PlaywrightDriver
		driver = PlaywrightDriver(
			browser_type=playwright_browser,
			headless=headless,
			viewport={"width": 1280, "height": 720},
		)
	return Browser(browser_profile=profile, driver=driver)


def _save_screenshot(step_number: int, screenshot_b64: str | None) -> None:
	if not screenshot_b64:
		return
	path = Path(SCREENSHOT_DIR)
	path.mkdir(parents=True, exist_ok=True)
	file_path = path / f"step_{step_number}.png"
	try:
		data = base64.b64decode(screenshot_b64)
		file_path.write_bytes(data)
	except Exception:
		pass


def _build_step_handler():
	recent_actions: list[str] = []

	def _handle_step(state_summary, agent_output, step_number: int) -> None:
		state = agent_output.current_state
		actions = []
		for action in agent_output.action:
			action_data = action.model_dump(exclude_unset=True)
			action_name = next(iter(action_data.keys())) if action_data else 'action'
			actions.append(action_name)

		if state.memory:
			print(f'Step {step_number}: {state.memory}', flush=True)
		if actions:
			print(f'Actions: {", ".join(actions)}', flush=True)
		if state.next_goal:
			print(f'Next goal: {state.next_goal}', flush=True)
		print('', flush=True)

		# Detect repeated actions (stuck) and store screenshot for GLM/OpenAI vision context
		if actions:
			recent_actions.append(','.join(actions))
			if len(recent_actions) > STUCK_REPEAT_THRESHOLD:
				recent_actions.pop(0)

			if len(recent_actions) == STUCK_REPEAT_THRESHOLD and len(set(recent_actions)) == 1:
				_save_screenshot(step_number, getattr(state_summary, 'screenshot', None))

	return _handle_step


async def plan_task(llm, task: str) -> str:
	"""Planner sub-agent: produce a concise plan."""
	try:
		system = SystemMessage(
			content=(
				"You are a planning sub-agent. Provide a concise, numbered step-by-step plan "
				"(5-10 steps) to accomplish the user task. Do not use browser actions; output plain text."
			)
		)
		user = UserMessage(content=task)
		response = await llm.ainvoke([system, user], output_format=None)
		if not response:
			print('Warning: Planner returned empty response', flush=True)
			return ""
		if not hasattr(response, 'completion'):
			print(f'Warning: Planner response missing completion attribute: {type(response)}', flush=True)
			return ""
		completion = response.completion
		if completion is None:
			return ""
		plan_text = str(completion).strip()
		if not plan_text:
			print('Warning: Planner returned empty plan', flush=True)
		return plan_text
	except Exception as e:
		import traceback
		print(f'Warning: Planner error: {type(e).__name__}: {e}', flush=True)
		# Only show traceback for unexpected errors
		if 'JSON' not in type(e).__name__ and 'json' not in str(e).lower():
			traceback.print_exc()
		return ""


def build_executor_instructions() -> str:
	return (
		"You are the execution sub-agent. Follow the plan and adapt if needed. "
		"If a login or manual step is required, call wait_for_user. "
		"Before any destructive action (checkout, delete, unsubscribe, remove), "
		"call confirm_destructive and proceed only if the user confirms with 'yes'. "
		"If an action fails, adapt and try an alternative approach."
	)


async def run_task(
	task: str,
	profile_config: dict,
	tools_builder=None,
	step_callback=None,
	task_id=None,
) -> bool:
	"""Run a task with the specified LLM provider.
	tools_builder: callable returning Tools, or callable(task_id) for web UI.
	step_callback: optional (state_summary, agent_output, step_number) -> None for web step display.
	task_id: optional; if set and tools_builder accepts one arg, tools_builder(task_id) is called.
	"""
	provider = profile_config.get("provider", "custom")
	model = profile_config.get("model")
	try:
		llm = get_llm(
			provider,
			model=model,
			temperature=profile_config.get("temperature"),
			max_completion_tokens=profile_config.get("max_completion_tokens"),
		)
		print(f'Using LLM: provider={provider}, model={model or "default"}', flush=True)
	except Exception as e:
		print(f'Error initializing LLM (provider={provider}, model={model}): {e}', flush=True)
		return False

	fallback_llm = None
	if profile_config.get("enable_fallback") and str(profile_config.get("provider", "")).lower() != 'custom':
		try:
			fallback_llm = ChatBrowserUse()
		except Exception:
			pass

	plan_text = ""
	if profile_config.get("enable_subagents"):
		try:
			plan_text = await plan_task(llm, task)
		except Exception as e:
			print(f'Planner failed: {e}', flush=True)

	# Build browser with error handling
	backend = profile_config.get("browser_backend", "cdp")
	playwright_browser = profile_config.get("playwright_browser", "chromium")
	print(f'Building browser: backend={backend}, playwright_browser={playwright_browser}', flush=True)
	try:
		browser = build_browser(
			persistent=profile_config.get("persistent_session", False),
			profile_dir=profile_config.get("profile_dir"),
			headless=profile_config.get("headless", False),
			wait_between_actions=profile_config.get("wait_between_actions", DEFAULT_ACTION_DELAY_SECONDS),
			proxy=profile_config.get("proxy"),
			browser_backend=backend,
			playwright_browser=playwright_browser,
		)
	except Exception as e:
		print(f'Error creating browser: {e}', flush=True)
		print('Make sure Chrome/Chromium is installed and accessible.', flush=True)
		return False

	if tools_builder is not None and task_id is not None:
		try:
			tools = tools_builder(task_id)
		except TypeError:
			tools = tools_builder()
	else:
		tools = (tools_builder or build_tools)()

	executor_task = task
	if plan_text:
		executor_task = f"{task}\n\nPlan:\n{plan_text}"

	step_handler = step_callback if step_callback is not None else _build_step_handler()

	agent = Agent(
		task=executor_task,
		llm=llm,
		browser=browser,
		tools=tools,
		fallback_llm=fallback_llm,
		extend_system_message=build_executor_instructions(),
		use_vision=True,
		initial_actions=[{'wait': {'seconds': profile_config.get("start_delay_seconds", DEFAULT_START_DELAY_SECONDS)}}],
		register_new_step_callback=step_handler,
	)

	try:
		await agent.run()
	except Exception as e:
		import traceback
		error_msg = str(e)
		error_type = type(e).__name__
		print(f'\n❌ Task execution failed: {error_type}: {error_msg}', flush=True)
		
		# Check for browser/CDP connection errors
		if 'JSONDecodeError' in error_type and 'Expecting value' in error_msg:
			# Check if it's a browser connection issue
			traceback_str = ''.join(traceback.format_exc())
			if 'cdp_url' in traceback_str or 'webSocketDebuggerUrl' in traceback_str or 'version_info.json' in traceback_str:
				print('\n🔧 Browser Connection Error:', flush=True)
				print('   The browser failed to start or connect via CDP (Chrome DevTools Protocol).', flush=True)
				print('\n💡 Possible solutions:', flush=True)
				print('   1. Make sure Chrome/Chromium is installed', flush=True)
				print('   2. Close any existing Chrome instances that might be blocking the port', flush=True)
				print('   3. Try running with headless=False to see browser window', flush=True)
				print('   4. Check if another process is using the CDP port', flush=True)
				print('   5. Restart your computer if the issue persists', flush=True)
				print('\n📋 Full traceback:', flush=True)
				traceback.print_exc()
			else:
				# It's likely an LLM API error
				print('\n📋 Full traceback for debugging:', flush=True)
				traceback.print_exc()
				print('\n💡 Tip: This error often occurs when:', flush=True)
				print('   - LLM API returned empty or invalid response', flush=True)
				print('   - API key is invalid or expired', flush=True)
				print('   - Network issues or API timeout', flush=True)
				print('   - Try using a different provider or check API keys in .env', flush=True)
		# Print traceback for other JSON/parsing errors
		elif any(keyword in error_type or keyword in error_msg.lower() 
		       for keyword in ['JSON', 'json', 'Expecting value', 'Parse', 'parse', 'Decode']):
			print('\n📋 Full traceback for debugging:', flush=True)
			traceback.print_exc()
			print('\n💡 Tip: This error often occurs when:', flush=True)
			print('   - LLM API returned empty or invalid response', flush=True)
			print('   - API key is invalid or expired', flush=True)
			print('   - Network issues or API timeout', flush=True)
			print('   - Try using a different provider or check API keys in .env', flush=True)
		return False
	return True


def _print_queue_items(items: list[dict]) -> None:
	if not items:
		print("Queue is empty.", flush=True)
		return
	for item in items:
		print(
			f"- [{item.get('status')}] {item.get('task')} "
			f"(priority={item.get('priority')}, retries_left={item.get('retries_left')}, profile={item.get('profile')})",
			flush=True,
		)


def add_queue_item(profiles: dict, queue: dict) -> None:
	task = _prompt_str("Task").strip()
	if not task:
		print("Task is required", flush=True)
		return
	priority = _prompt_int("Priority (higher runs first)", 0) or 0
	retries = _prompt_int("Retries", 0) or 0
	profile_name = select_profile(profiles, "Select profile for this task")
	item = {
		"id": str(uuid4()),
		"task": task,
		"priority": priority,
		"retries_left": retries,
		"attempts": 0,
		"status": "pending",
		"profile": profile_name,
		"created_at": datetime.utcnow().isoformat(),
	}
	queue["items"].append(item)
	save_queue(queue)
	print("Task added to queue.", flush=True)


def clear_completed(queue: dict) -> None:
	queue["items"] = [i for i in queue["items"] if i.get("status") not in {"done", "failed"}]
	save_queue(queue)
	print("Cleared completed/failed items.", flush=True)


async def run_queue(profiles: dict, queue: dict) -> None:
	if queue.get("paused"):
		resume = _prompt_bool("Queue is paused. Resume now?", True)
		if not resume:
			return
		queue["paused"] = False
		save_queue(queue)

	while True:
		pending = [i for i in queue["items"] if i.get("status") == "pending"]
		if not pending:
			print("Queue is empty or all items are done.", flush=True)
			return
		pending.sort(key=lambda i: (-int(i.get("priority", 0)), i.get("created_at", "")))
		item = pending[0]
		item["status"] = "running"
		save_queue(queue)

		profile = get_profile_config(profiles, item.get("profile", DEFAULT_PROFILE_NAME))
		print(f"\nRunning task (priority={item.get('priority')}): {item.get('task')}", flush=True)
		success = await run_task(item["task"], profile)
		item["attempts"] = int(item.get("attempts", 0)) + 1

		if success:
			item["status"] = "done"
		else:
			if int(item.get("retries_left", 0)) > 0:
				item["retries_left"] = int(item.get("retries_left", 0)) - 1
				item["status"] = "pending"
				print("Task failed, will retry.", flush=True)
			else:
				item["status"] = "failed"
				print("Task failed, no retries left.", flush=True)

		save_queue(queue)

		next_action = _prompt_str("Press Enter to continue, 'p' to pause, 'q' to stop").lower()
		if next_action in {"p", "pause"}:
			queue["paused"] = True
			save_queue(queue)
			print("Queue paused.", flush=True)
			return
		if next_action in {"q", "quit", "stop"}:
			print("Stopped queue run.", flush=True)
			return


async def single_task_flow() -> None:
	profiles = load_profiles()
	profile_name = select_profile(profiles)
	profile_config = get_profile_config(profiles, profile_name)
	if _prompt_bool("Изменить параметры для этого запуска?", False):
		profile_name = select_profile(profiles, "Профиль")
		profile_config = get_profile_config(profiles, profile_name)
		provider = _prompt_str("Выбор LLM (custom/openai/glm)", profile_config.get("provider", "custom"))
		persistent_session = _prompt_bool("Persistent session", profile_config.get("persistent_session", False))
		enable_fallback = _prompt_bool("Fallback", profile_config.get("enable_fallback", False))
		enable_subagents = _prompt_bool("Sub-agent", profile_config.get("enable_subagents", False))
		profile_config = {
			**profile_config,
			"provider": provider,
			"persistent_session": persistent_session,
			"enable_fallback": enable_fallback,
			"enable_subagents": enable_subagents,
		}
	task = _prompt_str("Enter task").strip()
	if not task:
		print("Task is required", flush=True)
		return
	await run_task(task, profile_config)


async def batch_menu() -> None:
	profiles = load_profiles()
	queue = load_queue()
	while True:
		print("\nQueue menu:", flush=True)
		print("1) List queue", flush=True)
		print("2) Add task", flush=True)
		print("3) Run queue", flush=True)
		print("4) Pause queue", flush=True)
		print("5) Resume queue", flush=True)
		print("6) Clear completed/failed", flush=True)
		print("7) Back", flush=True)
		choice = input("Select option: ").strip()
		if choice == "1":
			_print_queue_items(queue["items"])
		elif choice == "2":
			add_queue_item(profiles, queue)
		elif choice == "3":
			await run_queue(profiles, queue)
		elif choice == "4":
			queue["paused"] = True
			save_queue(queue)
			print("Queue paused.", flush=True)
		elif choice == "5":
			queue["paused"] = False
			save_queue(queue)
			await run_queue(profiles, queue)
		elif choice == "6":
			clear_completed(queue)
		elif choice == "7":
			return
		else:
			print("Unknown option", flush=True)


async def main() -> None:
	load_dotenv()
	configure_logs()
	while True:
		print("\nTask Hunter:", flush=True)
		print("1) Run single task", flush=True)
		print("2) Batch queue", flush=True)
		print("3) Manage profiles", flush=True)
		print("4) Exit", flush=True)
		choice = input("Select option: ").strip()
		if choice == "1":
			await single_task_flow()
		elif choice == "2":
			await batch_menu()
		elif choice == "3":
			profile_menu()
		elif choice == "4":
			return
		else:
			print("Unknown option", flush=True)


if __name__ == '__main__':
	asyncio.run(main())
