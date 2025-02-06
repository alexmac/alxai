from contextvars import ContextVar
from dataclasses import dataclass
from typing import Type

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from alxai.anthropic.conv import oneshot_conv as anthropic_oneshot_conv
from alxai.anthropic.conv import usermsg as anthropic_usermsg
from alxai.openai.conv import oneshot_conv, usermsg


class NotGiven:
  pass


NOT_GIVEN = NotGiven()


@dataclass(kw_only=True)
class ConvContext:
  model: str
  oai_client: AsyncOpenAI
  ds_client: AsyncOpenAI | None = None
  anthropic_client: AsyncAnthropic | None = None
  perplexity_client: AsyncOpenAI | None = None
  xai_client: AsyncOpenAI | None = None
  temperature: float | NotGiven = NOT_GIVEN
  reasoning_effort: str | NotGiven = NOT_GIVEN


conv_context = ContextVar[ConvContext]('conv_context')


def get_conv_context() -> ConvContext:
  return conv_context.get()


def set_conv_context(ctx: ConvContext):
  conv_context.set(ctx)


async def oneshot[ResponseType](msg: str, response_format: Type[ResponseType] | None = None) -> ResponseType | str | None:
  ctx = get_conv_context()

  reasoning_effort = ctx.reasoning_effort if isinstance(ctx.reasoning_effort, str) else 'medium'
  temperature = ctx.temperature if isinstance(ctx.temperature, float) else 1

  try:
    if 'claude' in ctx.model:
      assert ctx.anthropic_client is not None
      msg += '\n\n DO NOT respond with any preamble, just pure JSON.'
      return await anthropic_oneshot_conv(
        ctx.anthropic_client,
        [anthropic_usermsg(msg)],
        response_format=response_format,
        model=ctx.model,
        temperature=temperature,
      )
    else:
      if 'deepseek' in ctx.model:
        client = ctx.ds_client
      else:
        client = ctx.oai_client
      assert client is not None

      return await oneshot_conv(
        client,
        [usermsg(msg)],
        response_format=response_format,
        reasoning_effort=reasoning_effort,  # type: ignore
        model=ctx.model,
      )
  except Exception:
    return None
