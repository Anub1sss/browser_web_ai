import os
from typing import TYPE_CHECKING

from browser_ai.logging_config import setup_logging

# Only set up logging if not in MCP mode or if explicitly requested
if os.environ.get('BROWSER_USE_SETUP_LOGGING', 'true').lower() != 'false':
	from browser_ai.config import CONFIG

	# Get log file paths from config/environment
	debug_log_file = getattr(CONFIG, 'BROWSER_USE_DEBUG_LOG_FILE', None)
	info_log_file = getattr(CONFIG, 'BROWSER_USE_INFO_LOG_FILE', None)

	# Set up logging with file handlers if specified
	logger = setup_logging(debug_log_file=debug_log_file, info_log_file=info_log_file)
else:
	import logging

	logger = logging.getLogger('browser_use')

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
	"""Patched __del__ that handles closed event loops without throwing noisy red-herring errors like RuntimeError: Event loop is closed"""
	try:
		# Check if the event loop is closed before calling the original
		if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
			# Event loop is closed, skip cleanup that requires the loop
			return
		_original_del(self)
	except RuntimeError as e:
		if 'Event loop is closed' in str(e):
			# Silently ignore this specific error
			pass
		else:
			raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


# Type stubs for lazy imports - fixes linter warnings
if TYPE_CHECKING:
	from browser_ai.agent.prompts import SystemPrompt
	from browser_ai.agent.service import Agent

	from browser_ai.agent.views import ActionModel, ActionResult, AgentHistoryList
	from browser_ai.browser import BrowserProfile, BrowserSession
	from browser_ai.browser import BrowserSession as Browser
	from browser_ai.dom.service import DomService
	from browser_ai.llm import models
	from browser_ai.llm.anthropic.chat import ChatAnthropic
	from browser_ai.llm.azure.chat import ChatAzureOpenAI
	from browser_ai.llm.browser_use.chat import ChatBrowserUse
	from browser_ai.llm.google.chat import ChatGoogle
	from browser_ai.llm.groq.chat import ChatGroq
	from browser_ai.llm.mistral.chat import ChatMistral
	from browser_ai.llm.oci_raw.chat import ChatOCIRaw
	from browser_ai.llm.ollama.chat import ChatOllama
	from browser_ai.llm.openai.chat import ChatOpenAI
	from browser_ai.llm.vercel.chat import ChatVercel
	from browser_ai.tools.service import Controller, Tools

	# Lazy imports mapping - only import when actually accessed
_LAZY_IMPORTS = {
	'Agent': ('browser_ai.agent.service', 'Agent'),
	# System prompt (moderate weight due to agent.views imports)
	'SystemPrompt': ('browser_ai.agent.prompts', 'SystemPrompt'),
	# Agent views (very heavy - over 1 second!)
	'ActionModel': ('browser_ai.agent.views', 'ActionModel'),
	'ActionResult': ('browser_ai.agent.views', 'ActionResult'),
	'AgentHistoryList': ('browser_ai.agent.views', 'AgentHistoryList'),
	'BrowserSession': ('browser_ai.browser', 'BrowserSession'),
	'Browser': ('browser_ai.browser', 'BrowserSession'),  # Alias for BrowserSession
	'BrowserProfile': ('browser_ai.browser', 'BrowserProfile'),
	# Tools (moderate weight)
	'Tools': ('browser_ai.tools.service', 'Tools'),
	'Controller': ('browser_ai.tools.service', 'Controller'),  # alias
	# DOM service (moderate weight)
	'DomService': ('browser_ai.dom.service', 'DomService'),
	# Chat models (very heavy imports)
	'ChatOpenAI': ('browser_ai.llm.openai.chat', 'ChatOpenAI'),
	'ChatGoogle': ('browser_ai.llm.google.chat', 'ChatGoogle'),
	'ChatAnthropic': ('browser_ai.llm.anthropic.chat', 'ChatAnthropic'),
	'ChatBrowserUse': ('browser_ai.llm.browser_use.chat', 'ChatBrowserUse'),
	'ChatGroq': ('browser_ai.llm.groq.chat', 'ChatGroq'),
	'ChatMistral': ('browser_ai.llm.mistral.chat', 'ChatMistral'),
	'ChatAzureOpenAI': ('browser_ai.llm.azure.chat', 'ChatAzureOpenAI'),
	'ChatOCIRaw': ('browser_ai.llm.oci_raw.chat', 'ChatOCIRaw'),
	'ChatOllama': ('browser_ai.llm.ollama.chat', 'ChatOllama'),
	'ChatVercel': ('browser_ai.llm.vercel.chat', 'ChatVercel'),
	# LLM models module
	'models': ('browser_ai.llm.models', None),
}


def __getattr__(name: str):
	"""Lazy import mechanism - only import modules when they're actually accessed."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			if attr_name is None:
				# For modules like 'models', return the module itself
				attr = module
			else:
				attr = getattr(module, attr_name)
			# Cache the imported attribute in the module's globals
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'Agent',
	'BrowserSession',
	'Browser',  # Alias for BrowserSession
	'BrowserProfile',
	'Controller',
	'DomService',
	'SystemPrompt',
	'ActionResult',
	'ActionModel',
	'AgentHistoryList',
	# Chat models
	'ChatOpenAI',
	'ChatGoogle',
	'ChatAnthropic',
	'ChatBrowserUse',
	'ChatGroq',
	'ChatMistral',
	'ChatAzureOpenAI',
	'ChatOCIRaw',
	'ChatOllama',
	'ChatVercel',
	'Tools',
	'Controller',
	# LLM models module
	'models',
]
