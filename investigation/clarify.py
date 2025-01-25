from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field

from alxai.openai.convclass import ConvClass, usermsg
from investigation.investigation import Investigation, InvestigationConv


class Clarifications(BaseModel):
  clarifications: List[str] = Field(description='Questions we need the user to answer before we can use the AWS CLI tool.')


@dataclass(kw_only=True)
class Clarify(InvestigationConv):
  async def response(self, msg: Clarifications) -> Optional['ConvClass']:
    print(msg.model_dump_json(indent=2))
    return None


async def clarify_question(client, investigation: Investigation):
  prompt = f"""A user is asking us a question that requires us to gather data from the AWS cli tool to answer it. Before we do that are there any clarifying questions we should ask the user before using the tool. Here is the question:

{investigation.prompt}

Respond with a JSON object that conforms to the JSON schema {Clarifications.model_json_schema()}."""

  await Clarify(client=client, messages=[usermsg(prompt)], investigation=investigation, model='o1-mini', response_format=Clarifications).run()
