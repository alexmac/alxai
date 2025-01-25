import asyncio
import logging
import sys

from alxai.openai.client import get_openai_client
from investigation.are_we_done import are_we_done
from investigation.gather_data import gather_data
from investigation.gather_naabu import gather_naabu
from investigation.investigation import Investigation
from investigation.summarize_result import summarize_result


async def prototype_aws(client, log):
  prompt = 'Find my running EC2 instances and do a port scan against their internal IPs'
  # prompt = 'list all of the certificates used by my load balancers and show me when they expire'
  investigation = Investigation(prompt=prompt)

  # await gather_docs(client, investigation)

  while True:
    try:
      await gather_data(client, investigation)
      await gather_naabu(client, investigation)
    except RuntimeError:
      continue

    if await are_we_done(client, investigation):
      break

  result = await summarize_result(client, investigation)
  investigation._save_master_index()
  print(f'\n\n\n### Final Summary:\n{result}')


async def main():
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  httpx_log = logging.getLogger('httpx')
  httpx_log.setLevel(logging.WARNING)
  log = logging.getLogger()

  async with get_openai_client() as client:
    # , get_deepseek_client() as deepseek_client:
    await prototype_aws(client, log)


if __name__ == '__main__':
  asyncio.run(main())
