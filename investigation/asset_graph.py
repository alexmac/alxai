from typing import Dict, List

from pydantic import BaseModel

AssetID = str


class AssetNode(BaseModel):
  asset_id: AssetID
  asset_name: str
  asset_type: str


class AssetEdge(BaseModel):
  source_id: AssetID
  target_id: AssetID
  edge_type: str


class AssetGraph(BaseModel):
  nodes: Dict[AssetID, AssetNode]
  edges: Dict[AssetID, List[AssetEdge]]

  def add_node(self, node: AssetNode) -> None:
    if node.asset_id not in self.nodes:
      self.nodes[node.asset_id] = node

  def add_edge(self, edge: AssetEdge) -> None:
    if edge.source_id not in self.edges:
      self.edges[edge.source_id] = []

    for e in self.edges[edge.source_id]:
      if e.target_id == edge.target_id:
        return

    self.edges[edge.source_id].append(edge)

  def update(self, asset_graph: 'AssetGraph') -> None:
    self.nodes.update(asset_graph.nodes)
    for edges in asset_graph.edges.values():
      for edge in edges:
        self.add_edge(edge)

  def to_gml(self) -> str:
    gml_string = 'graph [\n'
    for node in self.nodes.values():
      gml_string += f'  node [\n    id {node.asset_id}\n    label {node.asset_name}\n  ]\n'
    return gml_string
