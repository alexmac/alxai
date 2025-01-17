import asyncio
import logging
import sys
from typing import List

from openai.types.chat import ParsedChatCompletionMessage
from pydantic import BaseModel

from alxai.openai.client import get_openai_client
from alxai.openai.conv import Conv, start_conv, usermsg


class SecurityTask(BaseModel):
  description: str
  priority: int


class TaskProposals(BaseModel):
  tasks: List[SecurityTask]


class TaskDetail(BaseModel):
  more_details: str


async def response_handler(conv: Conv, message: ParsedChatCompletionMessage[TaskProposals]):
  user_response = input('Which task do you want more detail about? ')
  return conv.respond(f'tell me more about task: {user_response}', response_format=TaskDetail, msg_handler=more_detail_handler)


async def more_detail_handler(conv: Conv, message: ParsedChatCompletionMessage[TaskProposals]):
  log = logging.getLogger()
  assert message.parsed
  log.info(f'Detail: {message.parsed.model_dump_json(indent=2)}')


async def main():
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  # httpx_log = logging.getLogger('httpx')
  # httpx_log.setLevel(logging.WARNING)

  async with get_openai_client() as client:
    await start_conv(
      client,
      response_handler,
      [usermsg("""You are a cyber security expert. Propose five tasks that you would suggest doing to improve the posture of an organization. Respond with JSON.""")],
      response_format=TaskProposals,
      model='gpt-4o-mini',
    )


if __name__ == '__main__':
  asyncio.run(main())
