import json
import uuid
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field

from alxai.base.cli import CliError, run_cli
from alxai.openai.convclass import ConvClass, usermsg
from investigation.investigation import Investigation, InvestigationConv


class AWSCliToolArguments(BaseModel):
  command_arguments: List[str] = Field(description='A single AWS CLI command to run including the command name and any arguments. Do not quote the arguments. e.g. ["aws", "s3", "ls"].')


def prompt(investigation: Investigation) -> str:
  prompt = f"""# Goal
You are a cyber security, devops and infrastructure expert who focuses on conducting investigations into cloud infrastructure environments. You are tasked with proposing AWS cli commands one by one to run (no bash scripting allowed) that will gather additional information to help answer the following question: "{investigation.prompt}"

# Approach
- Propose a single command each time.
- Each command should do one very specific thing, do not try to merge tasks.
- You can use the output from previously run commands as input to this cli command.
- The command should be a single call to the "aws" cli tool.
- prefer "--output json" over "--output text"
- Some commands require ARNs, make sure to suggest commands that will find ARNs before commands that need to use them.

# Response
Respond with a JSON object that conforms to the JSON schema {json.dumps(AWSCliToolArguments.model_json_schema(), indent=2)}.
"""
  if investigation.files:
    prompt += f'\n# Commands run so far\n{investigation.summarize_files()}'
  return prompt


@dataclass(kw_only=True)
class GatherData(InvestigationConv):
  failure_count: int = 0

  async def response(self, msg: AWSCliToolArguments) -> Optional['ConvClass']:
    args = msg.command_arguments
    try:
      stdout, _ = await run_cli(args)
    except CliError as e:
      self.failure_count += 1

      if self.failure_count > 1:
        raise RuntimeError(f'Failed to run AWS CLI command after 3 attempts: {e}')

      return self.respond(
        f"""# Error
{e}
Please Fix the command and try again. Respond with a JSON object that conforms to the JSON schema {AWSCliToolArguments.model_json_schema()} No documentation or additional commentary. If the command failed because you didn't provide an ARN then it is likely you should run a different command that would find suitable ARNs."""
      )

    tool_id = uuid.uuid4()
    await self.investigation.add_file(self.client, stdout, f'aws_cli_output_{tool_id}', f'AWS CLI output for: {" ".join(args)}')

    return self.respond(
      f"""# command succeeded with output:
{stdout}"""
    )


async def gather_data(client, investigation: Investigation):
  await GatherData(client=client, messages=[usermsg(prompt(investigation))], investigation=investigation, model='o1-mini', response_format=AWSCliToolArguments).run()
