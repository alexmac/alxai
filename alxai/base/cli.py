import subprocess


class CliError(Exception):
  pass


def invoke_cli(args: list[str], expect_first_arg: str = '', unquote: bool = True) -> tuple[subprocess.CompletedProcess[str], list[str]]:
  if expect_first_arg and args[0] != expect_first_arg:
    args = [expect_first_arg] + args

  if unquote:
    for i, arg in enumerate(args):
      if arg.startswith('"') and arg.endswith('"'):
        args[i] = arg[1:-1]
      elif arg.startswith("'") and arg.endswith("'"):
        args[i] = arg[1:-1]

  print(f'🛠️ {" ".join(args)}')

  return subprocess.run(args, capture_output=True, text=True), args


async def run_cli(args: list[str], expect_first_arg: str = '') -> tuple[str, list[str]]:
  try:
    result, actual_args = invoke_cli(args, expect_first_arg=expect_first_arg)
    retcode, stdout, stderr = result.returncode, result.stdout, result.stderr
  except Exception as e:
    raise CliError(f'Unknown Error: {e}')

  if retcode != 0:
    if stderr:
      raise CliError(f'exited with code {retcode}: {stderr}')
    else:
      raise CliError(f'exited with code {retcode}')

  return stdout, actual_args
