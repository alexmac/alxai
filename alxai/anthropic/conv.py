import asyncio
import copy
import logging
import uuid
from abc import abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass
from logging import Logger
from time import time
from typing import Callable
from uuid import UUID

import anthropic
from anthropic.types import Message, MessageParam, ModelParam, TextBlockParam

from alxai.openai.tool import ToolExecutor

type MsgFailureHandler = Callable[['Conv', str, Message], Awaitable[Conv | None]]
type MsgHandler = Callable[['Conv', Message], Awaitable[Conv | None]]


def usermsg(msg: str) -> MessageParam:
  return MessageParam(
    role='user',
    content=[
      TextBlockParam(
        type='text',
        text=msg,
      ),
    ],
  )


def systemmsg(msg: str) -> MessageParam:
  return MessageParam(
    role='user',
    content=[
      TextBlockParam(
        type='text',
        text=msg,
      ),
    ],
  )


async def default_msg_handler(conv: 'Conv', message: Message) -> 'Conv | None':
  return


async def default_msg_failure_handler(conv: 'Conv', finish_reason: str, message: Message) -> 'Conv | None':
  conv._log.error(f'Conversation ended unexpectedly with: {finish_reason}')
  return


def parsedMsgToParam(msg: Message):
  return MessageParam(role=msg.role, content=msg.content)


class ConvListener:
  log: Logger

  def __init__(self, log: Logger):
    self.log: Logger = log

  @abstractmethod
  def before_run(self, conv_id: UUID, msgs: list[MessageParam]) -> None:
    pass

  @abstractmethod
  def after_run(self, conv_id: UUID, msg: Message) -> None:
    pass


class DefaultConvListener(ConvListener):
  def __init__(self, log: Logger):
    super().__init__(log)
    self.start_time: float = 0

  def before_run(self, conv_id: UUID, msgs: list[MessageParam]) -> None:
    self.start_time = time()
    for msg in msgs:
      self.log.info(f'{msg["role"]}: {msg.get("content")}')

  def after_run(self, conv_id: UUID, msg: Message) -> None:
    time_taken = time() - self.start_time
    self.log.info(f'{msg.role}: {msg.content} (took {time_taken:.2f}s)')


class StructuredOuputError(Exception):
  raw: str

  def __init__(self, raw: str):
    super().__init__()
    self.raw = raw


@dataclass
class Conv:
  client: anthropic.AsyncAnthropic
  messages: list[MessageParam]
  msg_handler: MsgHandler
  tools: list[ToolExecutor] | None
  _sem: asyncio.Semaphore
  _log: Logger
  _conv_id: UUID
  model: ModelParam
  temperature: float
  response_format: type | None
  msg_failure_handler: MsgFailureHandler = default_msg_failure_handler
  reasoning_effort: str = 'medium'
  _listener_msg_idx: int = 0
  _listener: ConvListener | None = None

  def clone(self, msgs: list[MessageParam]) -> 'Conv':
    return Conv(
      client=self.client,
      messages=msgs,
      msg_handler=self.msg_handler,
      _sem=self._sem,
      _log=self._log,
      model=self.model,
      reasoning_effort=self.reasoning_effort,
      temperature=self.temperature,
      msg_failure_handler=self.msg_failure_handler,
      _conv_id=self._conv_id,
      _listener_msg_idx=self._listener_msg_idx,
      _listener=self._listener,
      response_format=self.response_format,
      tools=self.tools,
    )

  def _clone_msgs(self) -> list[MessageParam]:
    msgs: list[MessageParam] = copy.deepcopy(self.messages)
    for m in msgs:
      tool_calls = m.get('tool_calls', [])
      if 'tool_calls' in m and tool_calls == []:
        del m['tool_calls']
    return msgs

  def append(self, msg: MessageParam) -> 'Conv':
    assert msg is not None
    msgs: list[MessageParam] = self._clone_msgs()
    msgs.append(msg)
    return self.clone(msgs)

  def respond(self, msg: str, msg_handler: MsgHandler | None = None, response_format: type | None = None) -> 'Conv':
    msgs: list[MessageParam] = self._clone_msgs()
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

  async def _after(self, msg: Message):
    if self._listener:
      self._listener.after_run(self._conv_id, msg)
      self._listener_msg_idx = len(self.messages)

  async def get_parsed_response[T](self, message: Message, response_format: type[T] | None) -> T | str | None:
    assert len(message.content) == 1
    txt = message.content[0].text or ''  # type: ignore  # pyright: ignore[reportAttributeAccessIssue]
    if txt.startswith('```json'):
      txt = txt[7:-3]
    txt = txt.strip()

    if response_format is not None:
      try:
        return response_format.model_validate_json(txt)  # type: ignore  # pyright: ignore[reportAttributeAccessIssue]
      except Exception as e:
        raise StructuredOuputError(txt) from e
    else:
      return txt

  async def run(self) -> None:
    await self._before()

    model: ModelParam = self.model

    response = await self.client.messages.create(model=model, max_tokens=4096, messages=self.messages, temperature=self.temperature)

    nc = self.append(parsedMsgToParam(response))
    await nc._after(response)

    if response.stop_reason == 'tool_use':
      assert False
    elif response.stop_reason in ['max_tokens', 'stop_sequence']:
      nc = await nc.msg_failure_handler(nc, response.stop_reason, response)
    else:
      nc = await nc.msg_handler(nc, response)

    if nc:
      return await nc.run()


async def oneshot_conv[ResponseType](
  client: anthropic.AsyncAnthropic,
  messages: list[MessageParam],
  response_format: type[ResponseType] | None = None,
  reasoning_effort: str = 'medium',
  tools: list[ToolExecutor] | None = None,
  sem: asyncio.Semaphore | None = None,
  log: Logger | None = None,
  conv_id: UUID | None = None,
  model: ModelParam = 'gpt-4o',
  temperature: float = 1,
  msg_failure_handler: MsgFailureHandler = default_msg_failure_handler,
  listener_msg_idx: int = 0,
  listener: ConvListener | None = None,
  debug: bool = True,
) -> ResponseType | str | None:
  log = log or logging.getLogger()

  result = {}

  async def result_handler(conv: Conv, message: Message) -> None:
    result['output'] = await conv.get_parsed_response(message, response_format)

  c = Conv(
    client=client,
    msg_handler=result_handler,
    messages=messages,
    _sem=sem or asyncio.Semaphore(4),
    _log=log,
    model=model,
    reasoning_effort=reasoning_effort,
    temperature=temperature,
    msg_failure_handler=msg_failure_handler,
    _conv_id=conv_id or uuid.uuid4(),
    _listener_msg_idx=listener_msg_idx,
    _listener=listener or (DefaultConvListener(log) if debug else None),
    response_format=response_format,
    tools=None,
  )
  await c.run()

  return result['output']
