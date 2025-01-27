import asyncio
import logging
import random
from abc import abstractmethod
from dataclasses import dataclass, field
from logging import Logger
from typing import Any, List

type ConvID = str


class ConvListener:
  log: Logger

  def __init__(self, log: Logger):
    self.log: Logger = log

  @abstractmethod
  def before_run(self, conv_id: ConvID, msgs: List) -> None:
    pass

  @abstractmethod
  def after_run(self, conv_id: ConvID, msg: Any) -> None:
    pass


def generate_conv_id() -> ConvID:
  return random.randbytes(3).hex()


@dataclass(kw_only=True)
class ConvClassBase:
  _log: Logger = field(default_factory=lambda: logging.getLogger())
  _sem: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(4))
  _conv_id: ConvID = field(default_factory=generate_conv_id)
  _listener_msg_idx: int = 0
  _listeners: List[ConvListener] = field(default_factory=lambda: [])
