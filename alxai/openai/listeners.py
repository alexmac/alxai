import os
from logging import Logger
from typing import Iterable, List

from openai.types.chat import ChatCompletionMessageParam, ParsedChatCompletionMessage

from alxai.base.generic_conv import ConvID, ConvListener
from alxai.debug import CURRENT_RUN_DIR


def _get_msg_text(msg: ChatCompletionMessageParam) -> str:
  content = msg.get('content')
  if isinstance(content, str):
    return content
  elif isinstance(content, Iterable):
    text = ''
    for c in content:
      if c.get('text'):
        text += c.get('text', '')
    return text
  else:
    return ''


class DefaultConvListener(ConvListener):
  def __init__(self, log: Logger):
    super().__init__(log)
    self.counter = 0

  def before_run(self, conv_id: ConvID, msgs: List[ChatCompletionMessageParam]) -> None:
    for msg in msgs:
      with open(os.path.join(CURRENT_RUN_DIR, f'{conv_id}_{self.counter}_{msg["role"]}.txt'), 'w') as f:
        f.write(_get_msg_text(msg))
      self.counter += 1

  def after_run(self, conv_id: ConvID, msg: ParsedChatCompletionMessage) -> None:
    with open(os.path.join(CURRENT_RUN_DIR, f'{conv_id}_{self.counter}_{msg.role}.txt'), 'w') as f:
      if isinstance(msg.content, str):
        f.write(msg.content)


class AgentPrintListener(ConvListener):
  def __init__(self, log: Logger):
    super().__init__(log)

  def before_run(self, conv_id: ConvID, msgs: List[ChatCompletionMessageParam]) -> None:
    for msg in msgs:
      print(f'🕵️ {conv_id}> {_get_msg_text(msg)[:100]}...')

  def after_run(self, conv_id: ConvID, msg: ParsedChatCompletionMessage) -> None:
    if isinstance(msg.content, str):
      print(f'🧠 {conv_id}> {msg.content[: 100]}...')
