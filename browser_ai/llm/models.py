"""
Convenient access to LLM models.

Supported providers: custom (ChatBrowserUse), openai (ChatOpenAI), glm (ChatGLM).

Usage:
    from browser_ai import llm
    model = llm.openai_gpt_4o
    model = llm.bu_latest
"""

import os
from typing import TYPE_CHECKING

from browser_ai.llm.browser_use.chat import ChatBrowserUse
from browser_ai.llm.openai.chat import ChatOpenAI

if TYPE_CHECKING:
	from browser_ai.llm.base import BaseChatModel

# Type stubs for IDE autocomplete
openai_gpt_4o: 'BaseChatModel'
openai_gpt_4o_mini: 'BaseChatModel'
openai_gpt_4_1_mini: 'BaseChatModel'
openai_o1: 'BaseChatModel'
openai_o3: 'BaseChatModel'
openai_o3_mini: 'BaseChatModel'
openai_o4_mini: 'BaseChatModel'
bu_latest: 'BaseChatModel'
bu_1_0: 'BaseChatModel'


def get_llm_by_name(model_name: str):
	"""
	Factory function to create LLM instances from string names with API keys from environment.

	Args:
	    model_name: String name like 'openai_gpt_4o', 'bu_latest', etc.

	Returns:
	    LLM instance with API keys from environment variables

	Raises:
	    ValueError: If model_name is not recognized
	"""
	if not model_name:
		raise ValueError('Model name cannot be empty')

	parts = model_name.split('_', 1)
	if len(parts) < 2:
		raise ValueError(f"Invalid model name format: '{model_name}'. Expected format: 'provider_model_name'")

	provider = parts[0]
	model_part = parts[1]

	# Convert underscores back to dots/dashes for actual model names
	if 'gpt_4_1_mini' in model_part:
		model = model_part.replace('gpt_4_1_mini', 'gpt-4.1-mini')
	elif 'gpt_4o_mini' in model_part:
		model = model_part.replace('gpt_4o_mini', 'gpt-4o-mini')
	elif 'gpt_4o' in model_part:
		model = model_part.replace('gpt_4o', 'gpt-4o')
	else:
		model = model_part.replace('_', '-')

	if provider == 'openai':
		api_key = os.getenv('OPENAI_API_KEY')
		return ChatOpenAI(model=model, api_key=api_key)

	elif provider == 'bu':
		model = f'bu-{model_part.replace("_", "-")}'
		api_key = os.getenv('BROWSER_USE_API_KEY')
		return ChatBrowserUse(model=model, api_key=api_key)

	else:
		raise ValueError(f"Unknown provider: '{provider}'. Available providers: openai, bu")


def __getattr__(name: str) -> 'BaseChatModel':
	if name == 'ChatOpenAI':
		return ChatOpenAI  # type: ignore
	elif name == 'ChatBrowserUse':
		return ChatBrowserUse  # type: ignore

	try:
		return get_llm_by_name(name)
	except ValueError:
		raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'ChatOpenAI',
	'ChatBrowserUse',
	'get_llm_by_name',
	'openai_gpt_4o',
	'openai_gpt_4o_mini',
	'openai_gpt_4_1_mini',
	'openai_o1',
	'openai_o3',
	'openai_o3_mini',
	'openai_o4_mini',
	'bu_latest',
	'bu_1_0',
]
