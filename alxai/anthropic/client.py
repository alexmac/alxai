import os

import anthropic
import pydantic


class AnthropicConfig(pydantic.BaseModel):
  secret: str


home_dir = os.path.expanduser('~')

_config: AnthropicConfig | None = None


def _get_config() -> AnthropicConfig:
  global _config
  if not _config:
    with open(os.path.join(home_dir, '.anthropic/config.json')) as f:
      _config = AnthropicConfig.model_validate_json(f.read())
  return _config


def get_anthropic_client(org: str | None = None) -> anthropic.AsyncAnthropic:
  cfg = _get_config()

  return anthropic.AsyncAnthropic(
    api_key=cfg.secret,
  )
