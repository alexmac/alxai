import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from alxai.listener_queue import ListenerQueue
from alxai.openai.conv import structured_oneshot
from alxai.openai.convclass import usermsg
from investigation.investigation import FileMetadata, Investigation


class FactualStatements(BaseModel):
  statements: list[str] = Field(description='A list of factual statements about the account being inspected.')


def prompt(investigation: Investigation, content: str) -> str:
  prompt = f"""# Goal
Your goal is to summarize the information contained in the output of the AWS cli tool into a set of factual statements about the account being inspected.

# Approach
- Be extremely concise, each string should be one clear, actionable sentence.
- Focus only on information that would be useful for handling future queries.
- Must be self contained and not require prior conytext to be fully understood.
- Written in declarative form (e.g., "AWS account 1234 has 10 EC2 instances").
- do not duplicate information from previous statements.
- prefer reporting on facts about what is observed, rather than observations of things that do not exist.
- refer to everything with a definite name/identifier.
- do not use relative references e.g. "the query" is bad "the ec2 describe-instances query" is good.

# Response
Respond with a JSON object that conforms to the JSON schema {json.dumps(FactualStatements.model_json_schema(), indent=2)}.
"""
  if investigation.facts:
    prompt += f'\n# Facts so far\n{["- " + fact for fact in sorted(investigation.facts)]}'

  prompt += f'\n# AWS cli output to analyze:\n{content}'
  return prompt


@dataclass(kw_only=True)
class ExtractFactsListener(ListenerQueue[FileMetadata]):
  investigation: Investigation
  client: Any

  async def process(self, fm: FileMetadata):
    if not fm.filename.startswith('aws_cli_output'):
      return

    filepath = self.investigation.dir / fm.filename
    with open(filepath, 'r') as f:
      content = f.read()

    try:
      facts = await structured_oneshot(self.client, [usermsg(prompt(self.investigation, content))], model='o3-mini', response_format=FactualStatements)
    except Exception as e:
      print(f'Error extracting facts: {e}')
      return

    self.investigation.facts.extend(facts.statements)
    self.investigation._save_master_index()
