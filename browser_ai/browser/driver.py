"""Cross-browser driver abstraction. Keeps same recognition (DOM + screenshot) while allowing Chromium/Firefox/WebKit via Playwright."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from browser_ai.browser.events import (
	ClickCoordinateEvent,
	ClickElementEvent,
	GoBackEvent,
	NavigateToUrlEvent,
	ScrollEvent,
	TypeTextEvent,
)
from browser_ai.browser.views import BrowserStateSummary, PageInfo, TabInfo
from browser_ai.dom.views import (
	EnhancedDOMTreeNode,
	EnhancedSnapshotNode,
	SerializedDOMState,
	SimplifiedNode,
)
from browser_ai.dom.views import NodeType

if TYPE_CHECKING:
	from browser_ai.browser.session import BrowserSession

logger = logging.getLogger(__name__)


class _PlaywrightSessionManager:
	"""Minimal session_manager stub for Playwright driver (no CDP targets)."""

	def __init__(self, session: BrowserSession) -> None:
		self.browser_session = session
		self.logger = session.logger
		self._url = 'about:blank'
		self._title = ''

	def _target(self) -> Any:
		from browser_ai.browser.session import Target
		return Target(
			target_id=PLAYWRIGHT_TARGET_ID,
			target_type='page',
			url=self._url,
			title=self._title or 'Unknown title',
		)

	def get_target(self, target_id: str) -> Any:
		if target_id == PLAYWRIGHT_TARGET_ID:
			return self._target()
		return None

	def get_all_page_targets(self) -> list[Any]:
		return [self._target()]

	def get_focused_target(self) -> Any:
		return self._target()

	def get_all_targets(self) -> dict[str, Any]:
		return {PLAYWRIGHT_TARGET_ID: self._target()}

	def get_all_target_ids(self) -> list[str]:
		return [PLAYWRIGHT_TARGET_ID]

	async def clear(self) -> None:
		"""No-op for stub (driver owns browser lifecycle)."""
		pass

# Sentinel target_id for Playwright-backed session (no CDP targets)
PLAYWRIGHT_TARGET_ID = 'playwright-1'


def _make_minimal_node(
	index: int,
	tag_name: str,
	attributes: dict[str, str],
	inner_text: str = '',
) -> EnhancedDOMTreeNode:
	"""Build a minimal EnhancedDOMTreeNode for Playwright-driven state (no CDP)."""
	return EnhancedDOMTreeNode(
		node_id=index,
		backend_node_id=index,
		node_type=NodeType.ELEMENT_NODE,
		node_name=tag_name,
		node_value=inner_text or '',
		attributes=attributes,
		is_scrollable=None,
		is_visible=True,
		absolute_position=None,
		target_id=PLAYWRIGHT_TARGET_ID,
		frame_id=None,
		session_id=None,
		content_document=None,
		shadow_root_type=None,
		shadow_roots=None,
		parent_node=None,
		children_nodes=[],
		ax_node=None,
		snapshot_node=EnhancedSnapshotNode(
			is_clickable=True,
			cursor_style='pointer',
			bounds=None,
			clientRects=None,
			scrollRects=None,
			computed_styles=None,
			paint_order=None,
			stacking_contexts=None,
		),
	)


@runtime_checkable
class BrowserDriver(Protocol):
	"""Protocol for browser backends: same state format and events, different execution (CDP vs Playwright)."""

	async def start(self, session: BrowserSession) -> None:
		"""Start browser and attach to session (e.g. subscribe to event_bus)."""
		...

	async def close(self) -> None:
		"""Close browser and release resources."""
		...

	async def get_browser_state_summary(
		self,
		include_screenshot: bool = True,
		include_recent_events: bool = False,
	) -> BrowserStateSummary:
		"""Return current page state in the same format as CDP path (DOM + screenshot)."""
		...

	def attach_handlers(self, session: BrowserSession) -> None:
		"""Subscribe to session.event_bus for actions (click, type, navigate, etc.)."""
		...


class PlaywrightDriver:
	"""Playwright-backed driver: Chromium, Firefox, WebKit with same DOM/screenshot format and event handling."""

	def __init__(
		self,
		browser_type: Literal['chromium', 'firefox', 'webkit'] = 'chromium',
		headless: bool = True,
		viewport: dict[str, int] | None = None,
	):
		if browser_type not in ('chromium', 'firefox', 'webkit'):
			raise ValueError(f'Invalid browser_type: {browser_type!r}. Must be chromium, firefox, or webkit.')
		self.browser_type = browser_type
		self.headless = headless
		self.viewport = viewport or {'width': 1280, 'height': 720}
		self._session: BrowserSession | None = None
		self._playwright: Any = None
		self._browser: Any = None
		self._context: Any = None
		self._page: Any = None
		self._index_to_selector: dict[int, str] = {}
		self._last_state: BrowserStateSummary | None = None

	async def start(self, session: BrowserSession) -> None:
		from playwright.async_api import async_playwright

		self._session = session
		self._playwright = await async_playwright().start()

		bt = getattr(self._playwright, self.browser_type, None)
		if bt is None:
			await self._playwright.stop()
			raise RuntimeError(f'Playwright does not support browser type: {self.browser_type!r}')

		logger.info('Launching Playwright %s (headless=%s)', self.browser_type, self.headless)
		try:
			self._browser = await bt.launch(headless=self.headless)
		except Exception as e:
			await self._playwright.stop()
			self._playwright = None
			raise RuntimeError(
				f'Failed to launch Playwright {self.browser_type}. '
				f'Run "python3 -m playwright install" to install browsers. Error: {e}'
			) from e

		self._context = await self._browser.new_context(viewport=self.viewport)
		self._page = await self._context.new_page()
		session.agent_focus_target_id = PLAYWRIGHT_TARGET_ID
		session.session_manager = _PlaywrightSessionManager(session)
		session.session_manager._url = self._page.url
		session.session_manager._title = await self._page.title()
		logger.info('Playwright driver started: %s v%s', self.browser_type, self._browser.version)

	async def close(self) -> None:
		if self._context:
			await self._context.close()
			self._context = None
		if self._browser:
			await self._browser.close()
			self._browser = None
		if self._playwright:
			await self._playwright.stop()
			self._playwright = None
		self._page = None
		self._session = None
		self._index_to_selector.clear()
		self._last_state = None

	def attach_handlers(self, session: BrowserSession) -> None:
		bus = session.event_bus
		bus.on(NavigateToUrlEvent, self._on_navigate)
		bus.on(GoBackEvent, self._on_go_back)
		bus.on(ClickElementEvent, self._on_click_element)
		bus.on(ClickCoordinateEvent, self._on_click_coordinate)
		bus.on(TypeTextEvent, self._on_type_text)
		bus.on(ScrollEvent, self._on_scroll)
		logger.debug('Playwright driver: event handlers attached')

	async def _on_navigate(self, event: NavigateToUrlEvent) -> None:
		if not self._page:
			return
		await self._page.goto(event.url, wait_until='domcontentloaded', timeout=30000)
		self._index_to_selector.clear()
		self._last_state = None
		if self._session and isinstance(self._session.session_manager, _PlaywrightSessionManager):
			mgr = self._session.session_manager
			mgr._url = self._page.url
			mgr._title = (await self._page.title()) or ''

	async def _on_go_back(self, event: GoBackEvent) -> None:
		if not self._page:
			return
		await self._page.go_back()
		self._index_to_selector.clear()
		self._last_state = None
		if self._session and isinstance(self._session.session_manager, _PlaywrightSessionManager):
			mgr = self._session.session_manager
			mgr._url = self._page.url
			mgr._title = (await self._page.title()) or ''

	async def _on_click_element(self, event: ClickElementEvent) -> None:
		selector = self._index_to_selector.get(event.node.backend_node_id)
		if not selector:
			logger.warning('PlaywrightDriver: no selector for index %s', event.node.backend_node_id)
			return
		try:
			await self._page.click(selector, timeout=5000)
		except Exception as e:
			logger.warning('PlaywrightDriver click failed: %s', e)
			raise
		self._last_state = None

	async def _on_click_coordinate(self, event: ClickCoordinateEvent) -> None:
		if not self._page:
			return
		await self._page.mouse.click(event.coordinate_x, event.coordinate_y)
		self._last_state = None

	async def _on_type_text(self, event: TypeTextEvent) -> None:
		if not event.node.backend_node_id or event.node.backend_node_id == 0:
			await self._page.keyboard.type(event.text, delay=50)
			return
		selector = self._index_to_selector.get(event.node.backend_node_id)
		if not selector:
			logger.warning('PlaywrightDriver: no selector for index %s', event.node.backend_node_id)
			return
		try:
			if event.clear:
				await self._page.fill(selector, event.text)
			else:
				await self._page.type(selector, event.text, delay=50)
		except Exception as e:
			logger.warning('PlaywrightDriver type failed: %s', e)
			raise
		self._last_state = None

	async def _on_scroll(self, event: ScrollEvent) -> None:
		if not self._page:
			return
		amount = getattr(event, 'amount', 100) or 100
		dx = dy = 0
		if event.direction == 'up':
			dy = -amount
		elif event.direction == 'down':
			dy = amount
		elif event.direction == 'left':
			dx = -amount
		elif event.direction == 'right':
			dx = amount
		if dx or dy:
			await self._page.mouse.wheel(dx, dy)
		self._last_state = None

	async def get_browser_state_summary(
		self,
		include_screenshot: bool = True,
		include_recent_events: bool = False,
	) -> BrowserStateSummary:
		if not self._page:
			raise RuntimeError('Playwright driver not started or page closed')
		# Collect interactive elements with stable indices and selectors via JS
		elements_js = """
		() => {
			const dataAttr = 'data-browser-ai-id';
			const interactive = document.querySelectorAll(
				'a, button, input, select, textarea, [role="button"], [role="link"], [role="menuitem"], [onclick], [tabindex]:not([tabindex="-1"])'
			);
			const out = [];
			interactive.forEach((el, i) => {
				const rect = el.getBoundingClientRect();
				if (rect.width < 1 || rect.height < 1) return;
				const tag = el.tagName.toLowerCase();
				const attrs = {};
				for (const a of el.attributes) {
					if (['id','name','type','value','placeholder','aria-label','title','role'].includes(a.name))
						attrs[a.name] = a.value;
				}
				const text = (el.innerText || '').slice(0, 200);
				const index = i + 1;
				el.setAttribute(dataAttr, String(index));
				const selector = `[${dataAttr}="${index}"]`;
				out.push({ index, tag, attrs, text, selector });
			});
			return out;
		}
		"""
		try:
			raw = await self._page.evaluate(elements_js)
		except Exception as e:
			logger.warning('PlaywrightDriver: element collection failed: %s', e)
			raw = []
		self._index_to_selector.clear()
		selector_map: dict[int, EnhancedDOMTreeNode] = {}
		simplified_children: list[SimplifiedNode] = []
		for item in raw:
			idx = item['index']
			tag = item.get('tag', 'div')
			attrs = item.get('attrs') or {}
			text = item.get('text') or ''
			sel = item.get('selector') or f'[data-browser-ai-id="{idx}"]'
			self._index_to_selector[idx] = sel
			node = _make_minimal_node(idx, tag, attrs, text)
			selector_map[idx] = node
			simplified_children.append(
				SimplifiedNode(original_node=node, children=[], should_display=True, is_interactive=True)
			)
		# Fake root so serializer has a tree
		root_node = _make_minimal_node(0, 'html', {}, '')
		root_simplified = SimplifiedNode(original_node=root_node, children=simplified_children, should_display=True, is_interactive=False)
		dom_state = SerializedDOMState(_root=root_simplified, selector_map=selector_map)
		url = self._page.url
		title = await self._page.title()
		screenshot_b64: str | None = None
		if include_screenshot:
			try:
				buf = await self._page.screenshot(type='png', full_page=False)
				screenshot_b64 = base64.b64encode(buf).decode('ascii')
			except Exception as e:
				logger.debug('PlaywrightDriver screenshot failed: %s', e)
		page_info = PageInfo(
			viewport_width=self.viewport.get('width', 1280),
			viewport_height=self.viewport.get('height', 720),
			page_width=self.viewport.get('width', 1280),
			page_height=self.viewport.get('height', 720),
			scroll_x=0,
			scroll_y=0,
			pixels_above=0,
			pixels_below=0,
			pixels_left=0,
			pixels_right=0,
		)
		tabs = [TabInfo(url=url, title=title, target_id=PLAYWRIGHT_TARGET_ID)]
		state = BrowserStateSummary(
			dom_state=dom_state,
			url=url,
			title=title,
			tabs=tabs,
			screenshot=screenshot_b64,
			page_info=page_info,
			pixels_above=0,
			pixels_below=0,
			browser_errors=[],
			is_pdf_viewer=False,
			recent_events=None,
			pending_network_requests=[],
			pagination_buttons=[],
			closed_popup_messages=[],
		)
		self._last_state = state
		# Keep Playwright session_manager stub in sync for get_current_page_url/title etc.
		if self._session and isinstance(self._session.session_manager, _PlaywrightSessionManager):
			self._session.session_manager._url = url
			self._session.session_manager._title = title
		return state
