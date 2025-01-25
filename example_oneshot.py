import asyncio
import logging
import sys
from typing import List

from pydantic import BaseModel

from alxai.openai.client import get_openai_client
from alxai.openai.conv import oneshot_conv, usermsg


class SecurityTask(BaseModel):
  description: str
  priority: int


class TaskProposals(BaseModel):
  tasks: List[SecurityTask]


async def main():
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  httpx_log = logging.getLogger('httpx')
  httpx_log.setLevel(logging.WARNING)
  log = logging.getLogger()

  async with get_openai_client() as client:
    tasks = await oneshot_conv(
      client,
      [usermsg("""You are a cyber security expert. Propose five tasks that you would suggest doing to improve the posture of an organization. Respond with JSON.""")],
      response_format=TaskProposals,
      model='gpt-4o-mini',
    )
    if isinstance(tasks, TaskProposals):
      log.info(f'Tasks: {tasks.model_dump_json(indent=2)}')


if __name__ == '__main__':
  asyncio.run(main())
