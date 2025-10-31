import asyncio
import copy
import json
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from logging import Logger
from typing import Any, Callable

from openai import AsyncOpenAI, Omit, omit
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ParsedChatCompletionMessage
from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_reasoning_effort import ChatCompletionReasoningEffort

from alxai.base.generic_conv import ConvClassBase, ConvID, ConvListener, generate_conv_id
from alxai.openai.listeners import AgentPrintListener, DefaultConvListener
from alxai.openai.tool import ToolExecutor, get_tool_descriptions

type MsgFailureHandler = Callable[['Conv', str, ParsedChatCompletionMessage[Any]], Awaitable['Conv | None']]
type MsgHandler = Callable[['Conv', ParsedChatCompletionMessage[Any]], Awaitable['Conv | None']]


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


async def default_msg_handler(conv: 'Conv', message: ParsedChatCompletionMessage[Any]) -> 'Conv | None':
  return


async def default_msg_failure_handler(conv: 'Conv', finish_reason: str, message: ParsedChatCompletionMessage[Any]) -> 'Conv | None':
  conv._log.error(f'Conversation ended unexpectedly with: {finish_reason}')
  return


def parsedMsgToParam(msg: ParsedChatCompletionMessage[Any]) -> ChatCompletionAssistantMessageParam:
  return ChatCompletionAssistantMessageParam(role=msg.role, content=msg.content, tool_calls=msg.tool_calls)  # pyright: ignore[reportArgumentType]


@dataclass
class Conv(ConvClassBase):
  client: AsyncOpenAI
  messages: list[ChatCompletionMessageParam]
  msg_handler: MsgHandler
  tools: list[ToolExecutor] | Omit
  model: str
  temperature: float | Omit
  response_format: type | Omit
  msg_failure_handler: MsgFailureHandler = default_msg_failure_handler
  reasoning_effort: ChatCompletionReasoningEffort = 'medium'

  def __post_init__(self):
    if not self._listeners:
      self._listeners.append(DefaultConvListener(self._log))
      self._listeners.append(AgentPrintListener(self._log))

  def clone(self, msgs: list[ChatCompletionMessageParam]) -> 'Conv':
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
      _listeners=self._listeners,
      response_format=self.response_format,
      tools=self.tools,
    )

  def _clone_msgs(self) -> list[ChatCompletionMessageParam]:
    msgs: list[ChatCompletionMessageParam] = copy.deepcopy(self.messages)
    for m in msgs:
      tool_calls = m.get('tool_calls', [])
      if 'tool_calls' in m and tool_calls == []:
        del m['tool_calls']
    return msgs

  def append(self, msg: ChatCompletionMessageParam) -> 'Conv':
    assert msg is not None
    msgs: list[ChatCompletionMessageParam] = self._clone_msgs()
    msgs.append(msg)
    return self.clone(msgs)

  def respond(self, msg: str, msg_handler: MsgHandler | None = None, response_format: type | None = None) -> 'Conv':
    msgs: list[ChatCompletionMessageParam] = self._clone_msgs()
    msgs.append(usermsg(msg))
    nc = self.clone(msgs)
    if msg_handler:
      nc.msg_handler = msg_handler
    if response_format is not None:
      nc.response_format = response_format
    return nc

  async def _before(self):
    for listener in self._listeners:
      listener.before_run(self._conv_id, self.messages[self._listener_msg_idx :])
    self._listener_msg_idx = len(self.messages)

  async def _after(self, msg: ParsedChatCompletionMessage[Any]):
    for listener in self._listeners:
      listener.after_run(self._conv_id, msg)
    self._listener_msg_idx = len(self.messages)

  async def get_parsed_response[T](self, message: ParsedChatCompletionMessage[Any], response_format: type[T] | None) -> T | str | None:
    if message.parsed is None:
      txt = message.content or ''
      if txt.startswith('```json'):
        txt = txt[7:-3]
      txt = txt.strip()
      if response_format is not None and response_format != omit:
        return response_format.model_validate_json(txt)  # type: ignore  # pyright: ignore[reportAttributeAccessIssue]
      else:
        return txt
    else:
      return message.parsed

  async def run(self) -> None:
    await self._before()

    temperature = self.temperature or omit
    response_format = self.response_format
    model = self.model
    reasoning_effort = omit

    if model == 'o1-mini':
      response_format = omit
      temperature = omit
    elif model == 'o1' or model == 'o3-mini':
      reasoning_effort = self.reasoning_effort
      temperature = omit

    if 'deepseek' in str(self.client.base_url):
      # print('Using DeepSeek model')
      model = 'deepseek-reasoner'
      response_format = omit
    elif 'perplexity' in str(self.client.base_url):
      # print('Using Perplexity model')
      model = 'sonar'
      response_format = omit
    elif 'x.ai' in str(self.client.base_url):
      # print('Using XAI model')
      model = 'grok-beta'
      response_format = omit
      reasoning_effort = omit

    response = await self.client.beta.chat.completions.parse(
      model=model, messages=self.messages, reasoning_effort=reasoning_effort, response_format=response_format, tools=get_tool_descriptions(self.tools), temperature=temperature
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
  messages: list[ChatCompletionMessageParam],
  tools: list[ToolExecutor] | None = None,
  sem: asyncio.Semaphore | None = None,
  log: Logger | None = None,
  conv_id: ConvID | None = None,
  model: str = 'gpt-4o',
  reasoning_effort: ChatCompletionReasoningEffort = 'medium',
  temperature: float | Omit = omit,
  msg_failure_handler: MsgFailureHandler = default_msg_failure_handler,
  listeners: list[ConvListener] | None = None,
  response_format: type | Omit | None = None,
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
    reasoning_effort=reasoning_effort,
    temperature=temperature,
    msg_failure_handler=msg_failure_handler,
    _conv_id=conv_id or generate_conv_id(),
    _listeners=listeners or [],
    response_format=response_format if response_format is not None else omit,
    tools=tools or omit,
  )
  await c.run()


async def oneshot_conv[ResponseType](
  client: AsyncOpenAI,
  messages: list[ChatCompletionMessageParam],
  response_format: type[ResponseType] | None = None,
  reasoning_effort: ChatCompletionReasoningEffort = 'medium',
  tools: list[ToolExecutor] | Omit | None = None,
  sem: asyncio.Semaphore | None = None,
  log: Logger | None = None,
  conv_id: ConvID | None = None,
  model: str = 'gpt-4o',
  temperature: float | Omit = omit,
  msg_failure_handler: MsgFailureHandler = default_msg_failure_handler,
  listeners: list[ConvListener] | None = None,
  debug: bool = True,
) -> ResponseType | str | None:
  log = log or logging.getLogger()

  result = {}

  async def result_handler(conv: Conv, message: ParsedChatCompletionMessage[ResponseType]) -> None:
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
    _conv_id=conv_id or generate_conv_id(),
    _listeners=listeners or [],
    response_format=response_format if response_format is not None else omit,
    tools=tools or omit,
  )
  await c.run()

  return result['output']


async def structured_oneshot[ResponseType](
  client: AsyncOpenAI,
  messages: list[ChatCompletionMessageParam],
  response_format: type[ResponseType],
  reasoning_effort: ChatCompletionReasoningEffort = 'medium',
  tools: list[ToolExecutor] | Omit | None = None,
  sem: asyncio.Semaphore | None = None,
  log: Logger | None = None,
  conv_id: ConvID | None = None,
  model: str = 'gpt-4o',
  temperature: float | Omit = omit,
  msg_failure_handler: MsgFailureHandler = default_msg_failure_handler,
  listeners: list[ConvListener] | None = None,
  debug: bool = True,
) -> ResponseType:
  log = log or logging.getLogger()

  result = {}

  async def result_handler(conv: Conv, message: ParsedChatCompletionMessage[ResponseType]) -> None:
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
    _conv_id=conv_id or generate_conv_id(),
    _listeners=listeners or [],
    response_format=response_format,
    tools=tools or omit,
  )
  await c.run()

  if isinstance(result['output'], response_format):
    return result['output']
  else:
    raise ValueError(f'Expected {response_format} but got {type(result["output"])}')
