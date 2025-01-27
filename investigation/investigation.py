import asyncio
import datetime
import json
import os
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from alxai.openai.conv import oneshot_conv, usermsg
from alxai.openai.convclass import ConvClass


class FileMetadata(BaseModel):
  filename: str
  file_type: str
  reason_created: Optional[str] = None
  file_schema: Optional[Dict[str, Any]] = None
  file_summary: Optional[str] = None
  command_args: Optional[list[str]] = None


async def summarize_file(client, reason: str, filepath: Path) -> str:
  with open(filepath, 'r') as f:
    content = f.read()

  prompt = f"""# Task: Analyze this - {reason}

# Goal
- extract the most relevant information to form a summary of what this output explains.
- if the output describes certain objects then be sure to explain how many there are and what their ID/ARN is exactly
- keep your summary concise. one paragraph with no formatting.

# Full command output:
{content}"""

  response = await oneshot_conv(client, [usermsg(prompt)], model='o1-mini')

  assert response is not None
  return response


OUTPUT_DIR = Path('output/investigations')


class Investigation(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)
  prompt: str
  summary: str = ''
  files: Dict[str, FileMetadata] = Field(default_factory=dict)
  new_files: asyncio.Event = Field(default_factory=asyncio.Event, exclude=True)
  done: asyncio.Event = Field(default_factory=asyncio.Event, exclude=True)
  dir: Path = Field(exclude=True, default_factory=Path)
  client: Any = Field(exclude=True, default=None)

  @classmethod
  def create(cls, prompt: str, client: Any = None):
    investigation = cls(prompt=prompt, client=client)
    investigation._create_random_directory()
    investigation._save_master_index()
    return investigation

  def _create_random_directory(self):
    random_dir_name = ''.join(random.choices(string.ascii_letters + string.digits, k=12))

    date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = f'{date_str}_{random_dir_name}'

    self.dir = OUTPUT_DIR / dir_name
    os.makedirs(self.dir, exist_ok=True)

  def _save_master_index(self):
    master_index_path = self.dir / 'master_index.json'
    with open(master_index_path, 'w') as f:
      f.write(self.model_dump_json(indent=2))

  def summarize_files(self) -> str:
    summary = []
    for file, metadata in self.files.items():
      if metadata.file_type == 'json':
        summary.append(f'- {metadata.reason_created}: {metadata.file_summary}')
      elif metadata.file_type == 'txt':
        summary.append(f' - {metadata.reason_created}: {metadata.file_summary}')
    return '\n'.join(summary)

  def file_dump(self) -> str:
    summary = []
    for file, metadata in self.files.items():
      if metadata.file_type == 'json':
        with open(self.dir / file, 'r') as f:
          contents = json.load(f)
        summary.append(f' * We ran the following command {metadata.reason_created}. And got the following data: {contents}')
      elif metadata.file_type == 'txt':
        with open(self.dir / file, 'r') as f:
          contents = f.read()
        summary.append(f' * We ran the following command {metadata.reason_created}. And got the following data: {contents}')
      else:
        assert False, f'Unsupported file type: {metadata.file_type}'
    return '\n\n'.join(summary)

  async def add_file(self, client, content: str, filename: str, reason: str = ''):
    file_type = 'txt'
    try:
      json.loads(content)
      file_type = 'json'
    except json.JSONDecodeError:
      pass

    print(f'🗄️ Adding file {filename} with type {file_type}')

    filename = f'{filename}.{file_type}'
    filepath = self.dir / filename
    with open(filepath, 'w') as f:
      f.write(content)
    metadata = FileMetadata(filename=filename, file_type=file_type, reason_created=reason, file_summary=await summarize_file(client, reason, filepath))
    self.files[filename] = metadata
    self._save_master_index()
    self.new_files.set()
    self.new_files.clear()


@dataclass(kw_only=True)
class InvestigationConv(ConvClass):
  investigation: Investigation
