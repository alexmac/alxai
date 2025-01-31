import json
import os
from pathlib import Path
from typing import Dict

import openai
import pydantic
import tiktoken


class OpenAICredentials(pydantic.BaseModel):
  secret: str


class OpenAIConfig(pydantic.BaseModel):
  orgs: Dict[str, OpenAICredentials]


home_dir = Path(os.path.expanduser('~'))

_configs: Dict[str, OpenAIConfig] = {}


def _get_config(cfg: str = 'open_ai') -> OpenAIConfig:
  global _configs

  if cfg in _configs:
    return _configs[cfg]

  def compute_config() -> OpenAIConfig:
    nonlocal cfg
    config_path = home_dir / f'.{cfg}/config.json'
    if not config_path.exists():
      return OpenAIConfig(orgs={'non_existent': OpenAICredentials(secret='abc')})
    with config_path.open('rt') as f:
      return OpenAIConfig.model_validate_json(f.read())

  _configs[cfg] = compute_config()
  return _configs[cfg]


def get_openai_client(org: str | None = None) -> openai.AsyncOpenAI:
  cfg = _get_config()
  if not org:
    org = list(cfg.orgs.keys())[0]

  return openai.AsyncOpenAI(api_key=cfg.orgs[org].secret, organization=org)


def get_deepseek_client(org: str | None = None) -> openai.AsyncOpenAI:
  cfg = _get_config('deepseek')
  if len(cfg.orgs) == 0:
    raise RuntimeError('You have to have at least one organization configured in the deepseek config file.')
  if not org:
    org = list(cfg.orgs.keys())[0]

  print(cfg.orgs[org].secret)
  return openai.AsyncOpenAI(api_key=cfg.orgs[org].secret, base_url='https://api.deepseek.com')


def get_perplexity_client(org: str | None = None) -> openai.AsyncOpenAI:
  cfg = _get_config('perplexity')
  if not org:
    org = list(cfg.orgs.keys())[0]

  return openai.AsyncOpenAI(api_key=cfg.orgs[org].secret, base_url='https://api.perplexity.ai')


async def get_embedding(oai: openai.AsyncOpenAI, json_data):
  input_text = json.dumps(json_data) if isinstance(json_data, dict) else json_data

  response = await oai.embeddings.create(
    input=input_text[:8000],
    model='text-embedding-3-small',
  )
  return response


def count_tokens(text: str, model: str = 'gpt-4o') -> int:
  if not tiktoken.model.MODEL_TO_ENCODING.get(model):
    model = 'gpt-4o'

  encoding = tiktoken.encoding_for_model(model)
  return len(encoding.encode(text))
