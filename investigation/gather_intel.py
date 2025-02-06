import json
import uuid
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from alxai.openai.client import get_perplexity_client
from alxai.openai.conv import oneshot_conv
from alxai.openai.convclass import ConvClass, usermsg
from investigation.investigation import Investigation, InvestigationConv


class SearchQuery(BaseModel):
  query: str = Field(description='A single search query to run on the internet.')


def prompt(investigation: Investigation) -> str:
  prompt = f"""# Goal
You are a cyber security, devops and infrastructure expert who focuses on conducting investigations into cloud infrastructure environments. You are tasked with proposing an internet search query that will gather additional information to help answer the following question: "{investigation.prompt}"

# Approach
- Propose a single query focused one one specific thing.
- Only propose a search for something that would materially help us understand how to answer the user's question.
- Make sure your query does not duplicate prior queries.

# Response
Respond with a JSON object that conforms to the JSON schema {json.dumps(SearchQuery.model_json_schema(), indent=2)}.
"""
  if investigation.files:
    prompt += f'\n# Commands run so far\n{investigation.summarize_files()}'

  if investigation.data_frames:
    prompt += f'\n# Data Frames acquired so far\n{investigation.summarize_data_frames()}'

  if len(investigation.assets.nodes) > 0:
    prompt += f'\n# Asset Graph:\n{investigation.assets.to_gml()}'

  return prompt


@dataclass(kw_only=True)
class GatherIntel(InvestigationConv):
  failure_count: int = 0

  async def response(self, msg: SearchQuery) -> Optional['ConvClass']:
    query = msg.query

    async with get_perplexity_client() as perplexity_client:
      query_result = await oneshot_conv(perplexity_client, [usermsg(query)], model='sonar')
      assert isinstance(query_result, str)

    tool_id = uuid.uuid4()
    file_prefix = f'internet_query_{tool_id}'
    await self.investigation.add_file(self.client, query_result, file_prefix, f'Internet Query: "{query}"')

    return None


async def gather_intel(client, investigation: Investigation):
  await GatherIntel(client=client, messages=[usermsg(prompt(investigation))], investigation=investigation, model='o3-mini', response_format=SearchQuery).run()
