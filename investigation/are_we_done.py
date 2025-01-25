from pydantic import BaseModel, Field

from alxai.openai.conv import oneshot_conv, usermsg
from investigation.investigation import Investigation


class AreWeDoneModel(BaseModel):
  we_are_done: bool = Field(description='Whether we have enough data to answer the question')


async def are_we_done(client, investigation: Investigation):
  prompt = f"""Here is the state of the investigation:

  {investigation.summarize_files()}

  Do we have enough data to answer this question, or should we continue to gather more data?

  {investigation.prompt}

  Respond with a JSON object that conforms to the following JSON Schema:

  {AreWeDoneModel.model_json_schema()}
  """

  done = await oneshot_conv(
    client,
    [usermsg(prompt)],
    model='gpt-4o-mini',
    response_format=AreWeDoneModel,
  )

  assert done is not None
  assert isinstance(done, AreWeDoneModel)
  return done.we_are_done
