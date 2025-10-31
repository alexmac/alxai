import asyncio
import os
from datetime import datetime

BASE_OUTPUT_DIR = 'output/conversations'
CURRENT_RUN_DIR = os.path.join(BASE_OUTPUT_DIR, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(CURRENT_RUN_DIR, exist_ok=True)


def get_results_from_tasks[RType](tasks: list[asyncio.Task[RType]]) -> list[RType]:
  results: list[RType] = []
  for t in tasks:
    try:
      p: RType = t.result()
      if p:
        results.append(p)
    except Exception:
      pass
  return results
