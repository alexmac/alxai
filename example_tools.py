import asyncio
import logging
import random
import sys

from openai.types.chat import ChatCompletionToolMessageParam, ParsedChatCompletionMessage
from pydantic import BaseModel, Field

from alxai.openai.client import get_openai_client
from alxai.openai.conv import Conv, start_conv, usermsg
from alxai.openai.tool import ToolExecutor


class LinearEquation(BaseModel):
  equation: str = Field(description="The linear equation to solve, use 'x' and 'y' as the variable names.")
  x: int
  y: int


class LinearEquationSolver(ToolExecutor):
  name = 'linear_equation_solver'
  description = 'Solve a linear equation with two variables.'
  parameters = LinearEquation

  async def invoke(self, tool_id: str, arguments) -> ChatCompletionToolMessageParam:
    arguments = LinearEquation.model_validate(arguments)
    log = logging.getLogger()
    log.info(f'Solving {arguments.equation} with x={arguments.x} and y={arguments.y}')

    # ☠️☠️☠️
    result = eval(arguments.equation, {'x': arguments.x, 'y': arguments.y})
    # ☠️☠️☠️

    if result == 0:
      msg = 'Correct! The result is 0'
    else:
      msg = f'Incorrect! The result is {result}. Please call the tool again with a different set of values.'

    return ChatCompletionToolMessageParam(
      role='tool',
      content=msg,
      tool_call_id=tool_id,
    )


async def response_handler(conv: Conv, message: ParsedChatCompletionMessage):
  return None


async def main():
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  httpx_log = logging.getLogger('httpx')
  httpx_log.setLevel(logging.WARNING)

  async with get_openai_client() as client:
    await start_conv(
      client,
      response_handler,
      [
        usermsg(
          f"""Propose a simple linear equation using python expression syntax with two variables and no equals sign. Call the linear equation tool verifier with the equation and two values that should result in the equation equalling zero. Keep calling the tool until you are correct. Use a number larger than {random.randint(1, 100)} for the value of x."""
        )
      ],
      model='gpt-4o-mini',
      tools=[LinearEquationSolver()],
    )


if __name__ == '__main__':
  asyncio.run(main())
