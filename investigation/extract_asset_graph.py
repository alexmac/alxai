from dataclasses import dataclass
from typing import Any, List

from pydantic import BaseModel, Field

from alxai.listener_queue import ListenerQueue
from alxai.openai.conv import oneshot_conv, usermsg
from investigation.asset_graph import AssetGraph
from investigation.investigation import FileMetadata, Investigation


class ExtractPrimaryIdsModel(BaseModel):
  primary_ids: List[str] = Field(description='A list of IDs (ARNs or other types).')


def prompt(investigation: Investigation, content: str):
  return f"""# Goal:
Extract an asset graph from the provided AWS cli output. Ensure that all assets referenced by ID in the output are represented in the Graph, even if they have no known edges associated with them.

# Instructions:
- Use full ARNs as the asset_id for each asset node, do not confuse the name with the id.
- Only if an ARN is not used by a specific AWS resource should youuse the AWS resource name as the asset_id.
- The asset_name should be a short. human readable name for the asset.
- the edge_type should be a short single verb.
- The asset_type should be a single noun describing the aws object type.

# Response:
Respond with a JSON object that conforms to the following JSON Schema: {AssetGraph.model_json_schema()}

# CLI Output to analyze:
{content}
"""


@dataclass(kw_only=True)
class AssetGraphListener(ListenerQueue[FileMetadata]):
  investigation: Investigation
  client: Any

  async def process(self, fm: FileMetadata):
    if not fm.filename.startswith('aws_cli_output'):
      return

    filepath = self.investigation.dir / fm.filename
    with open(filepath, 'r') as f:
      content = f.read()

    try:
      asset_graph = await oneshot_conv(self.client, [usermsg(prompt(self.investigation, content))], model='o3-mini')
      asset_graph = AssetGraph.model_validate_json(asset_graph or '')
    except Exception as e:
      print(f'Error extracting asset graph: {e}')
      return

    self.investigation.assets.update(asset_graph)
    self.investigation._save_master_index()
