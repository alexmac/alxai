import asyncio
import logging
import sys

from alxai.base.context import ConvContext, set_conv_context
from alxai.openai.client import get_openai_client
from investigation.are_we_done import AreWeDoneListener
from investigation.extract_asset_graph import AssetGraphListener
from investigation.extract_facts import ExtractFactsListener
from investigation.gather_data import gather_data
from investigation.investigation import Investigation
from investigation.summarize_as_html import save_investigation_html
from investigation.summarize_result import SummarizeResultListener, summarize_result


async def gather_data_loop(client, investigation):
  while not investigation.done.is_set():
    try:
      await gather_data(client, investigation)
    except RuntimeError:
      continue


async def prototype_aws(client, log):
  prompt = 'Which ec2 instances can receive inbound SSH traffic from other hosts?'
  # prompt = 'list all of the certificates used by my load balancers and show me when they expire'
  # prompt = 'I have an ECS service called "cooltrans" that isn\'t working. What\'s wrong with it?'
  # prompt = 'list all my securityhub findings with a createdat in the last 10 days and summarize the high severity ones'
  investigation = Investigation.create(client=client, prompt=prompt)

  extract_facts_listener = ExtractFactsListener(investigation=investigation, client=client, done=investigation.done)
  investigation.add_listener(extract_facts_listener)

  asset_graph_listener = AssetGraphListener(investigation=investigation, client=client, done=investigation.done)
  investigation.add_listener(asset_graph_listener)

  summarize_result_listener = SummarizeResultListener(investigation=investigation, client=client, done=investigation.done)
  investigation.add_listener(summarize_result_listener)

  are_we_done_listener = AreWeDoneListener(investigation=investigation, client=client, done=investigation.done)
  investigation.add_listener(are_we_done_listener)

  # await gather_intel(client, investigation)

  await gather_data_loop(client, investigation)
  await investigation.shutdown()

  result = await summarize_result(client, investigation)
  print(f'\n\n\n### Final Summary:\n{result}')

  output_path = investigation.dir / 'index.html'
  save_investigation_html(investigation, output_path)


async def main():
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  httpx_log = logging.getLogger('httpx')
  httpx_log.setLevel(logging.WARNING)
  log = logging.getLogger()

  async with get_openai_client() as client:
    set_conv_context(
      ConvContext(
        model='o3-mini',
        reasoning_effort='medium',
        oai_client=client,
      )
    )
    await prototype_aws(client, log)


if __name__ == '__main__':
  asyncio.run(main())
