import json
import uuid
from dataclasses import dataclass
from typing import Optional

from alxai.openai.convclass import ConvClass, usermsg
from investigation.investigation import Investigation, InvestigationConv
from investigation.naabu_tool import NaabuToolArguments, run_naabu


@dataclass(kw_only=True)
class GatherData(InvestigationConv):
  failure_count: int = 0

  async def response(self, msg: NaabuToolArguments) -> Optional['ConvClass']:
    args = msg.command_arguments

    docs = self.investigation.docs.get(args[0] + ' ' + args[1])
    if not docs:
      docs = self.investigation.docs.get(args[1] + ' ' + args[2])
    if not docs:
      docs = '- no docs available'

    try:
      stdout, _ = await run_naabu(args)
    except Exception as e:
      self.failure_count += 1

      if self.failure_count > 1:
        raise RuntimeError(f'Failed to run CLI command after 3 attempts: {e}')

      return self.respond(
        f"""#Error running CLI command.
        
        # Docs
        {docs}

        Please Fix the command and try again. Respond with a JSON object that conforms to the JSON schema {NaabuToolArguments.model_json_schema()} No documentation or additional commentary."""
      )

    tool_id = uuid.uuid4()
    await self.investigation.add_file(self.client, stdout, f'aws_cli_output_{tool_id}', f'AWS CLI output for: {" ".join(args)}')


async def gather_naabu(client, investigation: Investigation):
  docs = '\n'.join([f'- {pair}\n```\n{docoutput}\n```' for pair, docoutput in investigation.docs.items()])

  prompt = f"""# Goal
You are a cyber security, devops and infrastructure expert who focuses on conducting investigations into cloud infrastructure environments. You are tasked with using naabu to answer the following question: "{investigation.prompt}"

# Approach
- The command should do one very specific thing, do not try to merge tasks.
- The input to naabu needs a hostname or ip address.
- Do not call it with CIDR notation, it will not work.
- Do not guess at IPs, they must come from the data you have gathered.
- If you are unable to meet these criteria then respond with an empty JSON object.

# Response
Respond with a JSON object that conforms to the JSON schema {json.dumps(NaabuToolArguments.model_json_schema(), indent=2)}.

# Latest Documentation
{docs}

# Commands run so far
{investigation.summarize_files()}"""

  print(prompt)

  await GatherData(client=client, messages=[usermsg(prompt)], investigation=investigation, model='o1-mini', response_format=NaabuToolArguments).run()
