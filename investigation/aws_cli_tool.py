import subprocess
from typing import List, Tuple

from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from pydantic import BaseModel, Field

from alxai.openai.tool import ToolExecutor
from investigation.investigation import Investigation


def invoke_aws_cli(args: List[str]) -> Tuple[int, str, str]:
  if args[0] != 'aws':
    args = ['aws'] + args

  for i, arg in enumerate(args):
    if arg.startswith('"') and arg.endswith('"'):
      args[i] = arg[1:-1]
    if arg.startswith("'") and arg.endswith("'"):
      args[i] = arg[1:-1]

  print(f'Running AWS CLI command: {" ".join(args)}')

  result = subprocess.run(args, capture_output=True, text=True)
  return result.returncode, result.stdout, result.stderr


async def run_aws_cli(args: List[str]) -> Tuple[str, List[str]]:
  try:
    retcode, stdout, stderr = invoke_aws_cli(args)
  except Exception as e:
    print(f'Error running AWS CLI command - {e}')
    raise ValueError(f'Error running AWS CLI command - {e}')

  if retcode != 0:
    raise ValueError(f'Error running AWS CLI command - {stderr}')

  return stdout, args


class AWSCliToolArguments(BaseModel):
  command_arguments: List[str] = Field(description='A single AWS CLI command to run including the command name and any arguments. Do not quote the arguments. e.g. ["aws", "s3", "ls"].')


class AWSCliTool(ToolExecutor):
  name = 'aws_cli'
  description = 'Run an AWS CLI command'
  parameters = AWSCliToolArguments

  def __init__(self, client, investigation: Investigation):
    super().__init__()
    self.client = client
    self.investigation = investigation

  async def invoke(self, tool_id: str, arguments) -> ChatCompletionToolMessageParam:
    args = AWSCliToolArguments.model_validate(arguments)

    try:
      stdout, actual_args = await run_aws_cli(args.command_arguments)
      await self.investigation.add_file(self.client, stdout, f'aws_cli_output_{tool_id}', f'AWS CLI output for: {" ".join(actual_args)}')
      return ChatCompletionToolMessageParam(tool_call_id=tool_id, role='tool', content=stdout)
    except Exception as e:
      print(f'Error running AWS CLI command - {e}')
      return ChatCompletionToolMessageParam(tool_call_id=tool_id, role='tool', content=f'Error: {e}')
