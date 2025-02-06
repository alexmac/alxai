import asyncio
import datetime
import json
import os
import random
import string
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from alxai.json import json_dumps
from alxai.openai.conv import oneshot_conv, usermsg
from alxai.openai.convclass import ConvClass
from investigation.asset_graph import AssetGraph


@dataclass(kw_only=True)
class ListenerQueue[MessageType]:
  queue: asyncio.Queue[MessageType] = field(default_factory=asyncio.Queue)
  done: asyncio.Event

  @abstractmethod
  async def process(self, msg: MessageType):
    pass

  async def _handle_done(self):
    await self.done.wait()
    self.queue.shutdown()

  async def run(self):
    done_task = asyncio.create_task(self._handle_done())

    try:
      while True:
        msg = await self.queue.get()
        await self.process(msg)
    except asyncio.QueueShutDown:
      pass
    finally:
      await done_task

