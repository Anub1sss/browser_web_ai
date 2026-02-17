import os
from typing import TYPE_CHECKING

from browser_ai.logging_config import setup_logging

# Only set up logging if not in MCP mode or if explicitly requested
if os.environ.get('BROWSER_USE_SETUP_LOGGING', 'true').lower() != 'false':
	from browser_ai.config import CONFIG

	debug_log_file = getattr(CONFIG, 'BROWSER_USE_DEBUG_LOG_FILE', None)
	info_log_file = getattr(CONFIG, 'BROWSER_USE_INFO_LOG_FILE', None)

	logger = setup_logging(debug_log_file=debug_log_file, info_log_file=info_log_file)
else:
	import logging

	logger = logging.getLogger('browser_use')

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
	try:
		if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
			return
		_original_del(self)
	except RuntimeError as e:
		if 'Event loop is closed' in str(e):
			pass
		else:
			raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


if TYPE_CHECKING:
	from browser_ai.agent.prompts import SystemPrompt
	from browser_ai.agent.service import Agent
	from browser_ai.agent.views import ActionModel, ActionResult, AgentHistoryList
	from browser_ai.browser import BrowserProfile, BrowserSession
	from browser_ai.browser import BrowserSession as Browser
	from browser_ai.dom.service import DomService
	from browser_ai.llm import models
	from browser_ai.llm.browser_use.chat import ChatBrowserUse
	from browser_ai.llm.openai.chat import ChatOpenAI
	from browser_ai.tools.service import Tools

_LAZY_IMPORTS = {
	'Agent': ('browser_ai.agent.service', 'Agent'),
	'SystemPrompt': ('browser_ai.agent.prompts', 'SystemPrompt'),
	'ActionModel': ('browser_ai.agent.views', 'ActionModel'),
	'ActionResult': ('browser_ai.agent.views', 'ActionResult'),
	'AgentHistoryList': ('browser_ai.agent.views', 'AgentHistoryList'),
	'BrowserSession': ('browser_ai.browser', 'BrowserSession'),
	'Browser': ('browser_ai.browser', 'BrowserSession'),
	'BrowserProfile': ('browser_ai.browser', 'BrowserProfile'),
	'Tools': ('browser_ai.tools.service', 'Tools'),
	'Controller': ('browser_ai.tools.service', 'Controller'),
	'DomService': ('browser_ai.dom.service', 'DomService'),
	'ChatOpenAI': ('browser_ai.llm.openai.chat', 'ChatOpenAI'),
	'ChatBrowserUse': ('browser_ai.llm.browser_use.chat', 'ChatBrowserUse'),
	'models': ('browser_ai.llm.models', None),
}


def __getattr__(name: str):
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module
			module = import_module(module_path)
			if attr_name is None:
				attr = module
			else:
				attr = getattr(module, attr_name)
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'Agent',
	'BrowserSession',
	'Browser',
	'BrowserProfile',
	'Controller',
	'DomService',
	'SystemPrompt',
	'ActionResult',
	'ActionModel',
	'AgentHistoryList',
	'ChatOpenAI',
	'ChatBrowserUse',
	'Tools',
	'Controller',
	'models',
]
