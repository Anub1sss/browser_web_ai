import asyncio
import time
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from browser_ai.llm.base import BaseChatModel
from browser_ai.llm.messages import BaseMessage, UserMessage, SystemMessage, AssistantMessage
from browser_ai.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


class ChatGLM(BaseChatModel):
	"""
	Client for GLM 4.6v via zai-sdk.
	
	Usage:
		from browser_ai.llm.glm.chat import ChatGLM
		llm = ChatGLM(model='glm-4.6v', api_key='your-api-key')
	"""

	def __init__(
		self,
		model: str = 'GLM-4.6V-Flash',
		api_key: str | None = None,
		temperature: float | None = None,
		timeout_seconds: float | None = None,
		max_retries: int = 3,
		retry_backoff_seconds: float = 2.0,
		**kwargs,
	):
		"""
		Initialize ChatGLM client.
		
		Args:
			model: Model name (default: 'GLM-4.6V-Flash')
			api_key: API key for zai-sdk. Defaults to ZAI_API_KEY env var.
			temperature: Temperature for generation
		"""
		try:
			from zai import ZaiClient
		except ImportError:
			raise ImportError(
				'zai-sdk is required for ChatGLM. Install it with: pip install zai-sdk'
			)
		
		self.model = model
		self.api_key = api_key
		self.temperature = temperature
		import os

		self.timeout_seconds = (
			timeout_seconds
			if timeout_seconds is not None
			else float(os.getenv('ZAI_TIMEOUT_SECONDS', '120'))
		)
		self.max_retries = max_retries
		self.retry_backoff_seconds = retry_backoff_seconds
		self._client = ZaiClient(api_key=api_key) if api_key else None
		
	@property
	def provider(self) -> str:
		return 'glm'
	
	@property
	def name(self) -> str:
		return self.model
	
	def _get_client(self):
		"""Get or create ZaiClient instance."""
		if self._client is None:
			import os
			from zai import ZaiClient
			
			api_key = self.api_key or os.getenv('ZAI_API_KEY')
			if not api_key:
				raise ValueError(
					'ZAI_API_KEY environment variable is required for GLM provider. '
					'Set it or pass api_key parameter.'
				)
			self._client = ZaiClient(api_key=api_key)
		return self._client
	
	def _extract_index_from_element_info(self, element_info: str) -> int | None:
		"""Extract index from element_info string like '[1]<textarea...' or '[123]' or just '[1]'."""
		import re
		if not element_info:
			return None
		# Convert to string if not already
		element_str = str(element_info).strip()
		# Match pattern [number] anywhere in the string
		match = re.search(r'\[(\d+)\]', element_str)
		if match:
			return int(match.group(1))
		return None
	
	def _transform_glm_response(self, parsed: dict[str, Any]) -> dict[str, Any]:
		"""Transform GLM response format to browser-use format.
		
		GLM sometimes returns actions with 'element_info' instead of 'index'.
		This function converts element_info like '[1]<textarea...' to index: 1.
		"""
		if 'action' not in parsed or not isinstance(parsed['action'], list):
			return parsed
		
		transformed_actions = []
		for action in parsed['action']:
			if not isinstance(action, dict):
				transformed_actions.append(action)
				continue
			
			# Find the action name and its params
			action_name = None
			action_params = None
			for key, value in action.items():
				if isinstance(value, dict):
					action_name = key
					action_params = value.copy()
					break
			
			# Handle simple forms like {'wait': 5}
			if not action_name and len(action) == 1:
				key, value = next(iter(action.items()))
				if key == 'wait' and isinstance(value, (int, float, str)):
					action_name = 'wait'
					action_params = {'seconds': value}
			
			if not action_name or action_params is None:
				transformed_actions.append(action)
				continue

			# If GLM returns empty click action, replace with a short wait
			if action_name == 'click' and not action_params:
				action_name = 'wait'
				action_params = {'seconds': 2}
			
			# Normalize wait action payloads (GLM uses timeout)
			if action_name == 'wait':
				if 'seconds' not in action_params:
					if 'timeout' in action_params:
						action_params['seconds'] = action_params.get('timeout')
					elif 'time' in action_params:
						action_params['seconds'] = action_params.get('time')
				action_params.pop('timeout', None)
				action_params.pop('time', None)
			
			# Transform actions that need index or coordinates
			needs_index = action_name in ['input', 'click', 'upload_file', 'dropdown_options', 'select_dropdown']
			
			if needs_index and 'index' not in action_params and 'coordinate_x' not in action_params:
				# GLM may use 'element_index', 'element_info', 'element', or 'box' instead of 'index' or coordinates
				if 'element_index' in action_params:
					action_params['index'] = action_params.get('element_index')
					action_params.pop('element_index', None)

				element_info = action_params.get('element_info') or action_params.get('element')
				if element_info:
					index = self._extract_index_from_element_info(str(element_info))
					if index is not None:
						action_params['index'] = index
					# Remove element_info/element as they're not part of the schema
					action_params.pop('element_info', None)
					action_params.pop('element', None)
				
				# Handle 'box' parameter (bounding box format: [[x1, y1, x2, y2]])
				if 'box' in action_params and action_name == 'click':
					box = action_params.get('box')
					if isinstance(box, list) and len(box) > 0:
						# box can be [[x1, y1, x2, y2]] or [x1, y1, x2, y2]
						box_coords = box[0] if isinstance(box[0], list) else box
						if len(box_coords) >= 4:
							# Calculate center of bounding box
							x1, y1, x2, y2 = box_coords[0], box_coords[1], box_coords[2], box_coords[3]
							action_params['coordinate_x'] = int((x1 + x2) / 2)
							action_params['coordinate_y'] = int((y1 + y2) / 2)
						elif len(box_coords) >= 2:
							# Use first two coordinates as x, y
							action_params['coordinate_x'] = int(box_coords[0])
							action_params['coordinate_y'] = int(box_coords[1])
					# Remove box as it's not part of the schema
					action_params.pop('box', None)
			
			# For input action, ensure 'clear' is set if not present
			if action_name == 'input' and 'clear' not in action_params:
				action_params['clear'] = True
			
			# Reconstruct action
			transformed_action = {action_name: action_params}
			transformed_actions.append(transformed_action)
		
		parsed['action'] = transformed_actions
		return parsed
	
	def _serialize_messages(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
		"""Convert BaseMessage list to zai-sdk format."""
		zai_messages = []
		
		for msg in messages:
			if isinstance(msg, SystemMessage):
				# Convert system message to user message with system role
				content = msg.text if isinstance(msg.content, str) else msg.text
				zai_messages.append({
					'role': 'system',
					'content': content
				})
			elif isinstance(msg, UserMessage):
				if isinstance(msg.content, str):
					zai_messages.append({
						'role': 'user',
						'content': msg.content
					})
				else:
					# Handle multi-part content (text + images)
					content_parts = []
					for part in msg.content:
						if part.type == 'text':
							content_parts.append({
								'type': 'text',
								'text': part.text
							})
						elif part.type == 'image_url':
							content_parts.append({
								'type': 'image_url',
								'image_url': {
									'url': part.image_url.url
								}
							})
					zai_messages.append({
						'role': 'user',
						'content': content_parts
					})
			elif isinstance(msg, AssistantMessage):
				content = msg.text if msg.content else ''
				zai_messages.append({
					'role': 'assistant',
					'content': content
				})
		
		return zai_messages
	
	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
	) -> ChatInvokeCompletion[str]: ...
	
	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any
	) -> ChatInvokeCompletion[T]: ...
	
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the GLM model with the given messages.
		
		Args:
			messages: List of chat messages
			output_format: Optional Pydantic model class for structured output
			
		Returns:
			Either a string response or an instance of output_format
		"""
		client = self._get_client()
		zai_messages = self._serialize_messages(messages)
		
		# Prepare request parameters
		request_params: dict[str, Any] = {
			'model': self.model,
			'messages': zai_messages,
		}
		
		if self.temperature is not None:
			request_params['temperature'] = self.temperature
		
		# Enable thinking for GLM models
		request_params['thinking'] = {'type': 'enabled'}
		request_params['timeout'] = self.timeout_seconds
		
		# Add response_format for structured output if needed
		if output_format is not None:
			from browser_ai.llm.schema import SchemaOptimizer
			import json
			
			# Create JSON schema from Pydantic model
			json_schema = SchemaOptimizer.create_optimized_json_schema(output_format)
			
			# zai-sdk might support response_format similar to OpenAI
			# Try to add it if the API supports it
			try:
				request_params['response_format'] = {
					'type': 'json_schema',
					'json_schema': {
						'name': 'agent_output',
						'strict': True,
						'schema': json_schema
					}
				}
			except Exception:
				# If response_format is not supported, we'll parse JSON from text response
				pass
		
		# Make the API call - zai-sdk is synchronous, so wrap in executor
		def _sync_request():
			try:
				import logging
				logger = logging.getLogger(__name__)
				logger.debug(f'GLM API request params: {request_params}')

				# Retry loop for transient errors (e.g., 502)
				for attempt in range(1, self.max_retries + 1):
					try:
						response = client.chat.completions.create(**request_params)
						logger.debug(f'GLM API full response (attempt {attempt}): {response}')
						return response
					except Exception as inner_exc:
						err_text = str(inner_exc)
						# Retry only on likely transient errors
						if '502' in err_text or 'Bad Gateway' in err_text or 'Server disconnected' in err_text:
							logger.warning(f'GLM API transient error (attempt {attempt}): {err_text}')
							if attempt < self.max_retries:
								time.sleep(self.retry_backoff_seconds * attempt)
								continue
						raise
			except Exception as e:
				import logging
				logger = logging.getLogger(__name__)
				error_msg = str(e)
				
				# Check for specific error types from zai-sdk
				if '429' in error_msg or 'Insufficient balance' in error_msg or 'no resource package' in error_msg:
					error_msg = 'GLM API: Insufficient balance or no resource package. Please recharge your account.'
				elif '401' in error_msg or 'Unauthorized' in error_msg:
					error_msg = 'GLM API: Invalid API key. Please check your ZAI_API_KEY.'
				elif '404' in error_msg:
					error_msg = 'GLM API: Model not found. Please check the model name.'
				
				logger.error(f'GLM API error: {error_msg}', exc_info=True)
				from browser_ai.llm.exceptions import ModelProviderError, ModelRateLimitError
				
				# Raise rate limit error for 429
				if '429' in str(e) or 'Insufficient balance' in str(e):
					raise ModelRateLimitError(message=error_msg, model=self.name) from e
				else:
					raise ModelProviderError(message=error_msg, model=self.name) from e
		
		loop = asyncio.get_event_loop()
		response = await loop.run_in_executor(None, _sync_request)
		
		# Extract response content
		if not response or not hasattr(response, 'choices') or not response.choices:
			from browser_ai.llm.exceptions import ModelProviderError
			raise ModelProviderError(
				message='Empty or invalid response from GLM API',
				model=self.name
			)
		
		message = response.choices[0].message
		content = ''
		
		# Handle content - could be string or dict
		if hasattr(message, 'content'):
			if isinstance(message.content, str):
				content = message.content
			elif isinstance(message.content, dict):
				content = message.content.get('text', '') or str(message.content)
			elif message.content is not None:
				content = str(message.content)
		
		# Handle structured output if needed
		if output_format is not None:
			import json
			import logging
			logger = logging.getLogger(__name__)
			
			# Try to parse JSON from response
			parsed = None
			try:
				# First, try direct JSON parse
				if isinstance(content, str):
					parsed = json.loads(content)
				elif isinstance(content, dict):
					parsed = content
				else:
					parsed = json.loads(str(content))
			except (json.JSONDecodeError, TypeError) as e:
				logger.debug(f'Direct JSON parse failed: {e}, trying to extract JSON from text')
				# Try to extract JSON from text (might be wrapped in markdown code blocks or text)
				import re
				# Try to find JSON in code blocks first
				json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
				if not json_match:
					# Try to find any JSON object
					json_match = re.search(r'\{.*\}', content, re.DOTALL)
				if json_match:
					try:
						parsed = json.loads(json_match.group(1) if json_match.lastindex else json_match.group())
					except json.JSONDecodeError:
						pass
			
			if parsed is None:
				from browser_ai.llm.exceptions import ModelProviderError
				raise ModelProviderError(
					message=f'Could not parse structured output from GLM response: {content[:200]}...',
					model=self.name
				)
			
			# Transform GLM response format to browser-use format
			logger.debug(f'GLM response before transformation: {parsed}')
			parsed = self._transform_glm_response(parsed)
			logger.debug(f'GLM response after transformation: {parsed}')
			
			# Validate and create output_format instance
			try:
				completion = output_format.model_validate(parsed)
			except Exception as e:
				logger.error(f'Failed to validate structured output: {e}, parsed: {parsed}')
				from browser_ai.llm.exceptions import ModelProviderError
				raise ModelProviderError(
					message=f'Failed to validate structured output: {e}. Response: {str(parsed)[:200]}',
					model=self.name
				) from e
		else:
			completion = content
		
		# Extract usage information if available
		usage = None
		if hasattr(response, 'usage') and response.usage:
			usage = ChatInvokeUsage(
				prompt_tokens=getattr(response.usage, 'prompt_tokens', 0),
				prompt_cached_tokens=None,
				prompt_cache_creation_tokens=None,
				prompt_image_tokens=None,
				completion_tokens=getattr(response.usage, 'completion_tokens', 0),
				total_tokens=getattr(response.usage, 'total_tokens', 0),
			)
		
		return ChatInvokeCompletion(
			completion=completion,
			thinking=None,
			redacted_thinking=None,
			usage=usage,
			stop_reason=None,
		)
