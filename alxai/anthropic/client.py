import os

import anthropic
import pydantic
from pathlib import Path


class AnthropicConfig(pydantic.BaseModel):
  secret: str


home_dir = Path(os.path.expanduser('~'))

_config: AnthropicConfig | None = None


def _get_config() -> AnthropicConfig:
  global _config
  if _config:
    return _config
  
  config_path = home_dir / '.anthropic/config.json'
  if not config_path.exists():
    return AnthropicConfig(secret='abc')
  
  with config_path.open('rt') as f:
      _config = AnthropicConfig.model_validate_json(f.read())
  return _config


def get_anthropic_client(org: str | None = None) -> anthropic.AsyncAnthropic:
  cfg = _get_config()

  return anthropic.AsyncAnthropic(
    api_key=cfg.secret,
  )
