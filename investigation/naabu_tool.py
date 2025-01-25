import subprocess
from typing import List, Tuple

from pydantic import BaseModel, Field


def invoke_naabu(args: List[str]) -> Tuple[int, str, str]:
  if args[0] != 'naabu':
    args = ['naabu'] + args

  for i, arg in enumerate(args):
    if arg.startswith('"') and arg.endswith('"'):
      args[i] = arg[1:-1]
    if arg.startswith("'") and arg.endswith("'"):
      args[i] = arg[1:-1]

  print(f'Running CLI command: {" ".join(args)}')

  result = subprocess.run(args, capture_output=True, text=True)
  return result.returncode, result.stdout, result.stderr


async def run_naabu(args: List[str]) -> Tuple[str, List[str]]:
  try:
    retcode, stdout, stderr = invoke_naabu(args)
  except Exception as e:
    print(f'Error running AWS CLI command - {e}')
    raise ValueError(f'Error running AWS CLI command - {e}')

  if retcode != 0:
    raise ValueError(f'Error running AWS CLI command - {stderr}')

  return stdout, args


class NaabuToolArguments(BaseModel):
  command_arguments: List[str] = Field(description='A single Naabu CLI command to run including the command name and any arguments. Do not quote the arguments.')
