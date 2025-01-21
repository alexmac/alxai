import asyncio
import copy
import json
import logging
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Awaitable, Callable, List, Optional, Type
from uuid import UUID

from openai import NOT_GIVEN, AsyncOpenAI, NotGiven
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ParsedChatCompletionMessage
from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam

from alxai.openai.tool import ToolExecutor, get_tool_descriptions

type MsgFailureHandler = Callable[['Conv', str, ParsedChatCompletionMessage], Awaitable[Optional[Conv]]]
type MsgHandler = Callable[['Conv', ParsedChatCompletionMessage], Awaitable[Optional[Conv]]]


def usermsg(msg: str) -> ChatCompletionUserMessageParam:
  return ChatCompletionUserMessageParam(
    role='user',
    content=[
      ChatCompletionContentPartTextParam(
        type='text',
        text=msg,
      ),
    ],
  )


def systemmsg(msg: str) -> ChatCompletionSystemMessageParam:
  return ChatCompletionSystemMessageParam(
    role='system',
    content=[
      ChatCompletionContentPartTextParam(
        type='text',
        text=msg,
      ),
    ],
  )


async def default_msg_handler(conv: 'Conv', message: ParsedChatCompletionMessage) -> Optional['Conv']:
  return


async def default_msg_failure_handler(conv: 'Conv', finish_reason: str, message: ParsedChatCompletionMessage) -> Optional['Conv']:
  conv._log.error(f'Conversation ended unexpectedly with: {finish_reason}')
  return


def parsedMsgToParam(msg: ParsedChatCompletionMessage):
  return ChatCompletionAssistantMessageParam(role=msg.role, content=msg.content, tool_calls=msg.tool_calls)  # type: ignore


class ConvListener:
  log: Logger

  def __init__(self, log: Logger):
    self.log: Logger = log

  @abstractmethod
  def before_run(self, conv_id: UUID, msgs: List[ChatCompletionMessageParam]) -> None:
    pass

  @abstractmethod
  def after_run(self, conv_id: UUID, msg: ParsedChatCompletionMessage) -> None:
    pass


class DefaultConvListener(ConvListener):
  def __init__(self, log: Logger):
    super().__init__(log)

  def before_run(self, conv_id: UUID, msgs: List[ChatCompletionMessageParam]) -> None:
    for msg in msgs:
      self.log.info(f'{msg["role"]}: {msg.get("content")}')

  def after_run(self, conv_id: UUID, msg: ParsedChatCompletionMessage) -> None:
    self.log.info(f'{msg.role}: {msg.content}')


@dataclass
class Conv:
  client: AsyncOpenAI
  messages: List[ChatCompletionMessageParam]
  msg_handler: MsgHandler
  tools: List[ToolExecutor] | NotGiven
  _sem: asyncio.Semaphore
  _log: Logger
  _conv_id: UUID
  model: str
  temperature: float | NotGiven
  response_format: Type | NotGiven
  msg_failure_handler: MsgFailureHandler = default_msg_failure_handler
  _listener_msg_idx: int = 0
  _listener: Optional[ConvListener] = None

  def clone(self, msgs: List[ChatCompletionMessageParam]) -> 'Conv':
    return Conv(
      client=self.client,
      messages=msgs,
      msg_handler=self.msg_handler,
      _sem=self._sem,
      _log=self._log,
      model=self.model,
      temperature=self.temperature,
      msg_failure_handler=self.msg_failure_handler,
      _conv_id=self._conv_id,
      _listener_msg_idx=self._listener_msg_idx,
      _listener=self._listener,
      response_format=self.response_format,
      tools=self.tools,
    )

  def _clone_msgs(self) -> List[ChatCompletionMessageParam]:
    msgs: List[ChatCompletionMessageParam] = copy.deepcopy(self.messages)
    for m in msgs:
      tool_calls = m.get('tool_calls', [])
      if 'tool_calls' in m and tool_calls == []:
        del m['tool_calls']
    return msgs

  def append(self, msg: ChatCompletionMessageParam) -> 'Conv':
    assert msg is not None
    msgs: List[ChatCompletionMessageParam] = self._clone_msgs()
    msgs.append(msg)
    return self.clone(msgs)

  def respond(self, msg: str, msg_handler: MsgHandler | None = None, response_format: Type | None = None) -> 'Conv':
    msgs: List[ChatCompletionMessageParam] = self._clone_msgs()
    msgs.append(usermsg(msg))
    nc = self.clone(msgs)
    if msg_handler:
      nc.msg_handler = msg_handler
    if response_format is not None:
      nc.response_format = response_format
    return nc

  async def _before(self):
    if self._listener:
      self._listener.before_run(self._conv_id, self.messages[self._listener_msg_idx :])
      self._listener_msg_idx = len(self.messages)

  async def _after(self, msg: ParsedChatCompletionMessage):
    if self._listener:
      self._listener.after_run(self._conv_id, msg)
      self._listener_msg_idx = len(self.messages)

  async def run(self) -> None:
    await self._before()

    response_format = self.response_format
    if self.model == 'o1-mini':
      response_format = NOT_GIVEN

    model = self.model
    if 'deepseek' in str(self.client.base_url):
      print('Using DeepSeek model')
      model = 'deepseek-reasoner'

    response = await self.client.beta.chat.completions.parse(
      model=model, messages=self.messages, response_format=response_format, tools=get_tool_descriptions(self.tools), temperature=self.temperature
    )
    choice = response.choices[0]
    assert choice
    nc = self.append(parsedMsgToParam(choice.message))
    await nc._after(choice.message)

    if choice.finish_reason == 'tool_calls':
      assert choice.message.tool_calls
      for tool_call in choice.message.tool_calls:
        tool_call_id = tool_call.id
        handled = False
        if isinstance(self.tools, list):
          for tool in self.tools:
            if tool.name == tool_call.function.name:
              tool_call_result = await tool.invoke(tool_call_id, json.loads(tool_call.function.arguments))
              nc = nc.append(tool_call_result)
              handled = True
              break
        if not handled:
          self._log.error(f'Tool call {tool_call_id} via tool {tool_call.function.name}({tool_call.function.arguments}) not handled')
          assert False
    elif choice.finish_reason in ['length', 'content_filter', 'function_call']:
      nc = await nc.msg_failure_handler(nc, choice.finish_reason, choice.message)
    else:
      nc = await nc.msg_handler(nc, choice.message)

    if nc:
      return await nc.run()


async def start_conv(
  client: AsyncOpenAI,
  msg_handler: MsgHandler,
  messages: List[ChatCompletionMessageParam],
  tools: List[ToolExecutor] | None = None,
  sem: Optional[asyncio.Semaphore] = None,
  log: Optional[Logger] = None,
  conv_id: Optional[UUID] = None,
  model: str = 'gpt-4o',
  temperature: float | NotGiven = NOT_GIVEN,
  msg_failure_handler: MsgFailureHandler = default_msg_failure_handler,
  listener_msg_idx: int = 0,
  listener: Optional[ConvListener] = None,
  response_format: Type | NotGiven | None = None,
  debug: bool = True,
):
  log = log or logging.getLogger()
  c = Conv(
    client=client,
    msg_handler=msg_handler,
    messages=messages,
    _sem=sem or asyncio.Semaphore(4),
    _log=log,
    model=model,
    temperature=temperature,
    msg_failure_handler=msg_failure_handler,
    _conv_id=conv_id or uuid.uuid4(),
    _listener_msg_idx=listener_msg_idx,
    _listener=listener or (DefaultConvListener(log) if debug else None),
    response_format=response_format if response_format is not None else NOT_GIVEN,
    tools=tools or NOT_GIVEN,
  )
  await c.run()


async def oneshot_conv[ResponseType](
  client: AsyncOpenAI,
  messages: List[ChatCompletionMessageParam],
  response_format: Type[ResponseType] | None = None,
  tools: List[ToolExecutor] | NotGiven | None = None,
  sem: Optional[asyncio.Semaphore] = None,
  log: Optional[Logger] = None,
  conv_id: Optional[UUID] = None,
  model: str = 'gpt-4o',
  temperature: float | NotGiven = NOT_GIVEN,
  msg_failure_handler: MsgFailureHandler = default_msg_failure_handler,
  listener_msg_idx: int = 0,
  listener: Optional[ConvListener] = None,
  debug: bool = True,
) -> ResponseType | str | None:
  log = log or logging.getLogger()

  result = {}

  async def result_handler(conv: Conv, message: ParsedChatCompletionMessage[ResponseType]) -> None:
    if message.parsed is None:
      txt = message.content or ''
      if txt.startswith('```json'):
        txt = txt[7:-3]
      txt = txt.strip()
      if response_format is not None and response_format != NOT_GIVEN:
        result['output'] = response_format.model_validate_json(txt)  # type: ignore
      else:
        result['output'] = txt
    else:
      result['output'] = message.parsed

  c = Conv(
    client=client,
    msg_handler=result_handler,
    messages=messages,
    _sem=sem or asyncio.Semaphore(4),
    _log=log,
    model=model,
    temperature=temperature,
    msg_failure_handler=msg_failure_handler,
    _conv_id=conv_id or uuid.uuid4(),
    _listener_msg_idx=listener_msg_idx,
    _listener=listener or (DefaultConvListener(log) if debug else None),
    response_format=response_format if response_format is not None else NOT_GIVEN,
    tools=tools or NOT_GIVEN,
  )
  await c.run()

  return result['output']
