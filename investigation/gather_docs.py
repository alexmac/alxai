import json
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional

from pydantic import BaseModel, Field

from alxai.openai.convclass import ConvClass, usermsg
from investigation.investigation import Investigation, InvestigationConv


class CommandPair(BaseModel):
  command: str = Field(description='The AWS cli command to run e.g. s3')
  subcommand: str = Field(description='The AWS cli subcommand to run e.g. ls')


class CommandPairs(BaseModel):
  command_pairs: list[CommandPair] = Field(description='A list of AWS cli command and subcommand pairs')


@dataclass(kw_only=True)
class GatherCommandPairs(InvestigationConv):
  failure_count: int = 0

  async def response(self, msg: CommandPairs) -> Optional['ConvClass']:
    print(msg.model_dump_json(indent=2))
    self.investigation.docs = await get_command_help(msg)


async def gather_docs(client, investigation: Investigation):
  prompt = f"""# Goal
You are a cyber security, devops and infrastructure expert who focuses on conducting investigations into cloud infrastructure environments. You are tasked with proposing AWS cli commands to run (no bash scripting allowed) that will gather one additional bit of information to help answer the following question: "{investigation.prompt}"

Before we answer the question for real you must propose a likey set of AWS cli command and subcommand pairs so that we can retrieve the most accurate documentation before starting.

Respond with a JSON object that conforms to the JSON schema {json.dumps(CommandPairs.model_json_schema(), indent=2)}."""

  await GatherCommandPairs(client=client, messages=[usermsg(prompt)], investigation=investigation, model='o1-mini', response_format=CommandPairs).run()


ANSI_ESCAPE_RE = re.compile(r'(?:\x1B[@-Z\\-_]|\x1B\[0-?]*[ -/]*[@-~])')


async def get_command_help(command_pairs: CommandPairs) -> Dict[str, str]:
  """
  Runs 'aws <command> <subcommand> help | cat' for each command pair and combines the output.
  Returns a markdown-formatted string with all documentation.
  """

  docs = {}
  for pair in command_pairs.command_pairs:
    cmd = f'aws {pair.command} {pair.subcommand} help | ansifilter'
    try:
      result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
      if result.returncode == 0:
        # Strip console control characters
        cleaned_output = result.stdout.encode('ascii', 'ignore').decode('ascii')
        docs[f'{pair.command} {pair.subcommand}'] = cleaned_output
      else:
        print(f'# aws {pair.command} {pair.subcommand}\n\n```\nError: {result.stderr}\n```')
    except Exception as e:
      print(f'# aws {pair.command} {pair.subcommand}\n\n```\nError: {str(e)}\n```')

  return docs
