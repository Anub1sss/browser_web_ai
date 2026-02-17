"""
LLM integration layer.

Supported providers: ChatBrowserUse (custom), ChatOpenAI, ChatGLM.
"""

from typing import TYPE_CHECKING

from browser_ai.llm.base import BaseChatModel
from browser_ai.llm.messages import (
	AssistantMessage,
	BaseMessage,
	SystemMessage,
	UserMessage,
)
from browser_ai.llm.messages import (
	ContentPartImageParam as ContentImage,
)
from browser_ai.llm.messages import (
	ContentPartRefusalParam as ContentRefusal,
)
from browser_ai.llm.messages import (
	ContentPartTextParam as ContentText,
)

if TYPE_CHECKING:
	from browser_ai.llm.browser_use.chat import ChatBrowserUse
	from browser_ai.llm.glm.chat import ChatGLM
	from browser_ai.llm.openai.chat import ChatOpenAI

_LAZY_IMPORTS = {
	'ChatBrowserUse': ('browser_ai.llm.browser_use.chat', 'ChatBrowserUse'),
	'ChatOpenAI': ('browser_ai.llm.openai.chat', 'ChatOpenAI'),
	'ChatGLM': ('browser_ai.llm.glm.chat', 'ChatGLM'),
}

_model_cache: dict[str, 'BaseChatModel'] = {}


def __getattr__(name: str):
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module
			module = import_module(module_path)
			return getattr(module, attr_name)
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	if name in _model_cache:
		return _model_cache[name]

	try:
		from browser_ai.llm.models import __getattr__ as models_getattr
		attr = models_getattr(name)
		_model_cache[name] = attr
		return attr
	except (AttributeError, ImportError):
		pass

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'BaseMessage',
	'UserMessage',
	'SystemMessage',
	'AssistantMessage',
	'ContentText',
	'ContentRefusal',
	'ContentImage',
	'BaseChatModel',
	'ChatOpenAI',
	'ChatBrowserUse',
	'ChatGLM',
]
