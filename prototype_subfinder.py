import asyncio
import logging
import subprocess
import sys

from alxai.openai.client import get_openai_client
from investigation.gather_subfinder import gather_subfinder
from investigation.investigation import Investigation
from investigation.summarize_result import summarize_result


async def prototype_aws(client, log):
  prompt = 'What can you tell me about 0xcafe.tech'
  # prompt = 'list all of the certificates used by my load balancers and show me when they expire'
  investigation = Investigation(prompt=prompt)

  # await gather_docs(client, investigation)
  subfinder_docs = subprocess.run('subfinder -h', shell=True, capture_output=True, text=True)
  subfinder_docs = subfinder_docs.stdout

  print(subfinder_docs)

  await gather_subfinder(client, investigation)

  result = await summarize_result(client, investigation)
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
