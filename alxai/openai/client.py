import json
import os
from typing import Dict

import openai
import pydantic


class OpenAICredentials(pydantic.BaseModel):
  secret: str


class OpenAIConfig(pydantic.BaseModel):
  orgs: Dict[str, OpenAICredentials]


home_dir = os.path.expanduser('~')

_configs: Dict[str, OpenAIConfig] = {}


def _get_config(cfg: str = 'open_ai') -> OpenAIConfig:
  global _configs
  if not _configs.get(cfg):
    with open(os.path.join(home_dir, f'.{cfg}/config.json')) as f:
      _configs[cfg] = OpenAIConfig.model_validate_json(f.read())
  return _configs[cfg]


def get_openai_client(org: str | None = None) -> openai.AsyncOpenAI:
  cfg = _get_config()
  if not org:
    org = list(cfg.orgs.keys())[0]

  return openai.AsyncOpenAI(api_key=cfg.orgs[org].secret, organization=org)


def get_deepseek_client(org: str | None = None) -> openai.AsyncOpenAI:
  cfg = _get_config('deepseek')
  if not org:
    org = list(cfg.orgs.keys())[0]

  print(cfg.orgs[org].secret)
  return openai.AsyncOpenAI(api_key=cfg.orgs[org].secret, base_url='https://api.deepseek.com')


async def get_embedding(oai: openai.AsyncOpenAI, json_data):
  input_text = json.dumps(json_data) if isinstance(json_data, dict) else json_data

  response = await oai.embeddings.create(
    input=input_text[:8000],
    model='text-embedding-3-small',
  )
  return response
