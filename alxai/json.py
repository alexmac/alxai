import datetime
import json
from typing import Any

import numpy as np


def json_serialize(obj: Any) -> str:
  if isinstance(obj, datetime.datetime):
    return obj.isoformat()
  elif isinstance(obj, np.ndarray):
    return obj.tolist()
  return obj


def json_dumps(obj: Any) -> str:
  return json.dumps(obj, default=json_serialize)
