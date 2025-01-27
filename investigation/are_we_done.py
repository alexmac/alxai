from pydantic import BaseModel, Field

from alxai.openai.conv import structured_oneshot, usermsg
from investigation.investigation import Investigation


class AreWeDoneModel(BaseModel):
  we_are_done: bool = Field(description='Whether we have enough data to answer the question')


def prompt(investigation: Investigation):
  return f"""# Here is the state of the investigation:

{investigation.summarize_files()}

Do we have enough data to answer the following question, or should we continue to gather more data?

"{investigation.prompt}"

Respond with a JSON object that conforms to the following JSON Schema: {AreWeDoneModel.model_json_schema()}
"""


async def are_we_done_task(client, investigation: Investigation):
  while True:
    await investigation.new_files.wait()
    done = await structured_oneshot(client, [usermsg(prompt(investigation))], model='gpt-4o-mini', response_format=AreWeDoneModel)
    if done.we_are_done:
      investigation.done.set()
      break
