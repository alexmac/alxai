from abc import abstractmethod
from typing import Any, Dict, List, Sequence, Type

from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel


class ToolExecutor:
  name: str
  description: str
  parameters: Type[BaseModel]

  @abstractmethod
  async def invoke(self, tool_id: str, arguments) -> ChatCompletionToolMessageParam:
    pass


def get_schema(t: ToolExecutor) -> Dict[str, Any]:
  schema = t.parameters.model_json_schema()
  schema['additionalProperties'] = False
  return schema


def get_tool_descriptions(tools: Sequence[ToolExecutor] | NotGiven | None) -> List[ChatCompletionToolParam] | NotGiven:
  if not tools or tools is NOT_GIVEN:
    return NOT_GIVEN
  oai_tool_descriptions = [ChatCompletionToolParam(type='function', function=FunctionDefinition(name=t.name, strict=True, description=t.description, parameters=get_schema(t))) for t in tools]
  assert oai_tool_descriptions
  return oai_tool_descriptions
