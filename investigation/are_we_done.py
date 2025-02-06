from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from alxai.listener_queue import ListenerQueue
from alxai.openai.conv import structured_oneshot, usermsg
from investigation.investigation import FileMetadata, Investigation


class AreWeDoneModel(BaseModel):
  we_are_done: bool = Field(description='Whether we have enough data to answer the question')


def prompt(investigation: Investigation):
  return f"""# Goal
Determine if the provided user question can be answered definitively using the data we have gathered so far. If more data should be gathered to answer it then respond "false", otherwise respond "true". The user questions is "{investigation.prompt}".

# Data gathered so far
{investigation.summarize_files()}

# Dataframes acquired so far
{investigation.summarize_data_frames()}

# Facts extracted so far
{investigation.summarize_facts()}

# Response
Respond with a JSON object that conforms to the following JSON Schema: {AreWeDoneModel.model_json_schema()}
"""


@dataclass(kw_only=True)
class AreWeDoneListener(ListenerQueue[FileMetadata]):
  investigation: Investigation
  client: Any

  async def process(self, fm: FileMetadata):
    done = await structured_oneshot(self.client, [usermsg(prompt(self.investigation))], model='o3-mini', response_format=AreWeDoneModel)
    if done.we_are_done:
      self.investigation.done.set()
