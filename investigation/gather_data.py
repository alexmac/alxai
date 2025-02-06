import json
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from alxai.base.cli import CliError, run_cli
from alxai.openai.client import count_tokens
from alxai.openai.conv import oneshot_conv
from alxai.openai.convclass import ConvClass, usermsg
from investigation.investigation import Investigation, InvestigationConv
from investigation.json_to_parquet import extract_dataframes_from_json
from investigation.summarize_as_html import save_investigation_html


class AWSCliToolArguments(BaseModel):
  command_arguments: List[str] = Field(description='A single AWS CLI command to run including the command name and any arguments. Do not quote the arguments. e.g. ["aws", "s3", "ls"].')


async def get_primary_id_key(client, data: Dict) -> str:
  result = await oneshot_conv(
    client, [usermsg(f'What is the primary id key for the provided data. respond only with a single string containing the exact key used, nothing else. Data: {json.dumps(data)}')], model='o3-mini'
  )
  assert isinstance(result, str)
  return result


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

As an example:
aws securityhub get-findings --filters '{{"CreatedAt":[{{"DateRange":{{"Value":10,"Unit":"DAYS"}}}}],"SeverityLabel":[{{"Value":"CRITICAL","Comparison":"EQUALS"}}]' --output json

# Response
Respond with a JSON object that conforms to the JSON schema {json.dumps(AWSCliToolArguments.model_json_schema(), indent=2)}.
"""
  if investigation.files:
    prompt += f'\n# Commands run so far\n{investigation.summarize_files()}'

  if investigation.data_frames:
    prompt += f'\n# Data Frames acquired so far\n{investigation.summarize_data_frames()}'

  if len(investigation.assets.nodes) > 0:
    prompt += f'\n# Asset Graph:\n{investigation.assets.to_gml()}'

  return prompt


@dataclass(kw_only=True)
class GatherData(InvestigationConv):
  failure_count: int = 0

  async def response(self, msg: AWSCliToolArguments) -> Optional['ConvClass']:
    args = msg.command_arguments
    try:
      stdout, _ = await run_cli(args, expect_first_arg='aws')
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
    file_prefix = f'aws_cli_output_{tool_id}'
    if count_tokens(stdout, self.model) > 10000:
      dfs = await extract_dataframes_from_json(json.loads(stdout), file_prefix, lambda data: get_primary_id_key(self.client, data))
      for name, df in dfs.items():
        await self.investigation.add_data_frame(self.client, df, f'{file_prefix}_{name}', f'AWS CLI output for: {" ".join(args)}')
    else:
      await self.investigation.add_file(self.client, stdout, file_prefix, f'AWS CLI output for: {" ".join(args)}')

      output_path = self.investigation.dir / 'index.html'
      save_investigation_html(self.investigation, output_path)

      if self.investigation.done.is_set():
        return None

      return self.respond(
        f"""# command succeeded with output:
  {stdout}"""
      )


async def gather_data(client, investigation: Investigation):
  await GatherData(client=client, messages=[usermsg(prompt(investigation))], investigation=investigation, model='o3-mini', response_format=AWSCliToolArguments).run()
