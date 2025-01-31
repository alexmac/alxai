import copy
import json
from dataclasses import dataclass
from typing import List, Optional, Self, Type

from openai import NOT_GIVEN, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ParsedChatCompletionMessage
from openai.types.chat.chat_completion_reasoning_effort import ChatCompletionReasoningEffort

from alxai.base.generic_conv import ConvClassBase
from alxai.model_quirks import strip_code_prefix
from alxai.openai.conv import parsedMsgToParam, usermsg
from alxai.openai.listeners import AgentPrintListener, DefaultConvListener
from alxai.openai.tool import ToolExecutor, get_tool_descriptions


@dataclass(kw_only=True)
class ConvClass[ResponseType](ConvClassBase):
  client: AsyncOpenAI
  messages: List[ChatCompletionMessageParam]
  tools: List[ToolExecutor] | None = None
  model: str = 'gpt-4o-mini'
  temperature: float | None = None
  response_format: Type[ResponseType] | None = None
  reasoning_effort: ChatCompletionReasoningEffort = 'low'

  def __post_init__(self):
    if not self._listeners:
      self._listeners.append(DefaultConvListener(self._log))
      self._listeners.append(AgentPrintListener(self._log))

  def _copy_msgs(self) -> List[ChatCompletionMessageParam]:
    msgs: List[ChatCompletionMessageParam] = copy.deepcopy(self.messages)
    for m in msgs:
      tool_calls = m.get('tool_calls', [])
      if 'tool_calls' in m and tool_calls == []:
        del m['tool_calls']
    return msgs

  def _append_msg(self, msg: ChatCompletionMessageParam) -> List[ChatCompletionMessageParam]:
    assert msg is not None
    msgs: List[ChatCompletionMessageParam] = self._copy_msgs()
    msgs.append(msg)
    return msgs

  @classmethod
  def respond_to(cls, conv: 'ConvClass', msg: str) -> 'ConvClass':
    msgs: List[ChatCompletionMessageParam] = conv._append_msg(usermsg(msg))
    nc = copy.copy(conv)
    nc.messages = msgs
    return nc

  def respond(self, msg: str) -> Self:
    return self.respond_via_msg(usermsg(msg))

  def respond_via_msg(self, msg: ChatCompletionMessageParam) -> Self:
    msgs: List[ChatCompletionMessageParam] = self._append_msg(msg)
    nc = copy.copy(self)
    nc.messages = msgs
    return nc

  async def _before(self):
    for listener in self._listeners:
      listener.before_run(self._conv_id, self.messages[self._listener_msg_idx :])
    self._listener_msg_idx = len(self.messages)

  async def _after(self, msg: ParsedChatCompletionMessage):
    for listener in self._listeners:
      listener.after_run(self._conv_id, msg)
    self._listener_msg_idx = len(self.messages)

  async def text_response(self, msg: str) -> Optional['ConvClass']:
    self._log.error(f'Conversation handler not implemented for text response: {msg}')
    return None

  async def response(self, msg: ResponseType) -> Optional['ConvClass']:
    self._log.error(f'Conversation handler not implemented for {msg}')
    return None

  async def failure(self, msg: ParsedChatCompletionMessage, finish_reason: str) -> Optional['ConvClass']:
    self._log.error(f'Conversation ended unexpectedly with: {finish_reason}')
    return None

  async def run(self) -> None:
    await self._before()

    temperature = self.temperature or NOT_GIVEN
    response_format = self.response_format or NOT_GIVEN
    model = self.model
    reasoning_effort = NOT_GIVEN

    if model == 'o1-mini':
      response_format = NOT_GIVEN
      temperature = NOT_GIVEN
    elif model == 'o1' or model == 'o3-mini':
      reasoning_effort = self.reasoning_effort
      temperature = NOT_GIVEN

    if 'deepseek' in str(self.client.base_url):
      # print('Using DeepSeek model')
      model = 'deepseek-reasoner'
      response_format = NOT_GIVEN
    elif 'perplexity' in str(self.client.base_url):
      # print('Using Perplexity model')
      model = 'sonar'
      response_format = NOT_GIVEN

    prev_role = None
    for m in self.messages:
      if m['role'] != prev_role:
        prev_role = m['role']
      else:
        assert False

    response = await self.client.beta.chat.completions.parse(
      model=model, messages=self.messages, reasoning_effort=reasoning_effort, response_format=response_format, tools=get_tool_descriptions(self.tools), temperature=temperature
    )
    choice = response.choices[0]
    assert choice
    nc = self.respond_via_msg(parsedMsgToParam(choice.message))
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
              nc = nc.respond_via_msg(tool_call_result)
              handled = True
              break
        if not handled:
          self._log.error(f'Tool call {tool_call_id} via tool {tool_call.function.name}({tool_call.function.arguments}) not handled')
          assert False
    elif choice.finish_reason in ['length', 'content_filter', 'function_call']:
      nc = await self.failure(choice.message, choice.finish_reason)
    else:
      if choice.message.parsed is None:
        txt = strip_code_prefix(choice.message.content or '')
        if self.response_format is not None and self.response_format != NOT_GIVEN:
          nc = await nc.response(self.response_format.model_validate_json(txt))  # type: ignore
        else:
          nc = await nc.text_response(txt)
      else:
        nc = await nc.response(choice.message.parsed)

    if nc:
      return await nc.run()
