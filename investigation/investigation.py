import datetime
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from alxai.openai.conv import oneshot_conv, usermsg
from alxai.openai.convclass import ConvClass


class FileMetadata(BaseModel):
  filename: str
  file_type: str
  reason_created: Optional[str] = None
  file_schema: Optional[Dict[str, Any]] = None
  file_summary: Optional[str] = None


class MasterIndex(BaseModel):
  prompt: str
  final_summary: Optional[str] = None
  files: Dict[str, FileMetadata] = Field(default_factory=dict)


async def summarize_file(client, reason: str, filepath: Path) -> str:
  with open(filepath, 'r') as f:
    content = f.read()

  prompt = f"""Here is the output of an AWS cli invocation. {reason}.

  Your goal is to pull out the most important bits of information so that you can respond with a short paragraph explaining what the data shows. If helpful use whitespace to make the output more readable.

  {content}
  """

  response = await oneshot_conv(client, [usermsg(prompt)], model='o1-mini')

  assert response is not None
  return response


class Investigation:
  OUTPUT_DIR = Path('output/investigations')

  def __init__(self, prompt: str, directory: Optional[str] = None):
    self.prompt = prompt
    self.docs: Dict[str, str] = {}
    if directory:
      self.dir = Path(directory)
      if not self.dir.exists():
        raise ValueError(f'Directory {directory} does not exist.')
    else:
      self.dir = self._create_random_directory()
    self.master_index_path = self.dir / 'master_index.json'
    if self.master_index_path.exists():
      self.master_index = MasterIndex.model_validate_json(self.master_index_path.read_text())
    else:
      self.master_index = MasterIndex(prompt=self.prompt)
      self._save_master_index()

  def _create_random_directory(self) -> Path:
    random_dir_name = ''.join(random.choices(string.ascii_letters + string.digits, k=12))

    date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    dir_name = f'{date_str}_{random_dir_name}'

    new_dir = self.OUTPUT_DIR / dir_name
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir

  def _save_master_index(self):
    with open(self.master_index_path, 'w') as f:
      f.write(self.master_index.model_dump_json(indent=2))

  def summarize_files(self) -> str:
    summary = []
    for file, metadata in self.master_index.files.items():
      if metadata.file_type == 'json':
        summary.append(f'- {metadata.reason_created}: {metadata.file_summary}')
      elif metadata.file_type == 'txt':
        summary.append(f' - {metadata.reason_created}: {metadata.file_summary}')
    return '\n'.join(summary)

  def file_dump(self) -> str:
    summary = []
    for file, metadata in self.master_index.files.items():
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

    filename = f'{filename}.{file_type}'
    filepath = self.dir / filename
    with open(filepath, 'w') as f:
      f.write(content)
    metadata = FileMetadata(filename=filename, file_type=file_type, reason_created=reason, file_summary=await summarize_file(client, reason, filepath))
    self.master_index.files[filename] = metadata
    self._save_master_index()

  def add_json(self, data: Any, filename: Optional[str] = None, reason: Optional[str] = None):
    filename = filename or f'data_{len(self.master_index.files) + 1}.json'
    filepath = self.dir / filename
    with open(filepath, 'w') as f:
      json.dump(data, f, indent=4)
    schema = list(data.keys()) if isinstance(data, dict) else None
    metadata = FileMetadata(filename=filename, file_type='json', reason_created=reason, file_schema={'keys': schema} if schema else None)
    self.master_index.files[filename] = metadata
    self._save_master_index()


@dataclass(kw_only=True)
class InvestigationConv(ConvClass):
  investigation: Investigation
