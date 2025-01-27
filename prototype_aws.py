import asyncio
import logging
import sys

from alxai.openai.client import get_deepseek_client, get_openai_client
from investigation.are_we_done import are_we_done_task
from investigation.gather_data import gather_data
from investigation.investigation import Investigation
from investigation.summarize_result import summarize_result


async def gather_data_loop(client, investigation):
  while True:
    try:
      await gather_data(client, investigation)
    except RuntimeError:
      continue


async def prototype_aws(client, log):
  # prompt = 'Which ec2 instances can receive inbound SSH on port 22?'
  # prompt = 'list all of the certificates used by my load balancers and show me when they expire'
  prompt = 'enumerate all my running ECS services and show me which ones are connected to public load balancers.'
  investigation = Investigation.create(client=client, prompt=prompt)

  done_task = asyncio.create_task(are_we_done_task(client, investigation))
  asyncio.create_task(gather_data_loop(client, investigation))

  await done_task

  result = await summarize_result(client, investigation)
  print(f'\n\n\n### Final Summary:\n{result}')


async def main():
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  httpx_log = logging.getLogger('httpx')
  httpx_log.setLevel(logging.WARNING)
  log = logging.getLogger()

  async with get_openai_client() as client, get_deepseek_client() as deepseek_client:
    await prototype_aws(client, log)


if __name__ == '__main__':
  asyncio.run(main())
