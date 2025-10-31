from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openai._types import Omit, omit
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel


@dataclass
class ToolExecutor:
  name: str
  description: str
  parameters: type[BaseModel]

  @abstractmethod
  async def invoke(self, tool_id: str, arguments: dict[str, Any]) -> ChatCompletionToolMessageParam:
    pass


def get_schema(t: ToolExecutor) -> dict[str, Any]:
  schema = t.parameters.model_json_schema()
  schema['additionalProperties'] = False
  return schema


def get_tool_descriptions(tools: Sequence[ToolExecutor] | Omit | None) -> list[ChatCompletionToolParam] | Omit:
  if not tools or tools is omit:
    return omit
  oai_tool_descriptions = [ChatCompletionToolParam(type='function', function=FunctionDefinition(name=t.name, strict=True, description=t.description, parameters=get_schema(t))) for t in tools]
  assert oai_tool_descriptions
  return oai_tool_descriptions
