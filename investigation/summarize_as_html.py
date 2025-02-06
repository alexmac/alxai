from pathlib import Path

import pandas as pd

from investigation.investigation import Investigation


def generate_vis_js_network(investigation: Investigation) -> str:
  nodes_js = []
  for node_id, node in investigation.assets.nodes.items():
    nodes_js.append(
      f"""{{
        id: "{node_id}",
        label: "{node.asset_name or node_id}",
        title: "{node.asset_type}: {node_id}"
      }}"""
    )

  edges_js = []
  for edges in investigation.assets.edges.values():
    for edge in edges:
      edges_js.append(
        f"""{{
        from: "{edge.source_id}",
        to: "{edge.target_id}",
        label: "{edge.edge_type}"
      }}"""
      )

  return f"""
  const nodes = new vis.DataSet([
    {',\n    '.join(nodes_js)}
  ]);

  const edges = new vis.DataSet([
    {',\n    '.join(edges_js)}
  ]);

  const container = document.getElementById('network');
  const data = {{
    nodes: nodes,
    edges: edges
  }};
  const options = {{}};
  const network = new vis.Network(container, data, options);
  """


def generate_dataframe_html(df: pd.DataFrame) -> list[str]:
  """Generate HTML fragments to display a pandas dataframe in a scrollable table."""
  html_fragments = []

  # Start scrollable container
  html_fragments.append('<div class="overflow-auto h-48">')
  html_fragments.append('<table class="min-w-full table-auto border-collapse">')

  # Header row
  html_fragments.append('<thead class="bg-gray-100 sticky top-0">')
  html_fragments.append('<tr>')
  for col in df.columns:
    html_fragments.append(f'<th class="px-4 py-2 border">{col}</th>')
  html_fragments.append('</tr>')
  html_fragments.append('</thead>')

  # Table body
  html_fragments.append('<tbody>')
  for _, row in df.iterrows():
    html_fragments.append('<tr class="hover:bg-gray-50">')
    for val in row:
      html_fragments.append(f'<td class="border min-w-16 max-w-48">{val}</td>')
    html_fragments.append('</tr>')
  html_fragments.append('</tbody>')

  html_fragments.append('</table>')
  html_fragments.append('</div>')

  return html_fragments


def generate_investigation_html(investigation: Investigation) -> str:
  """Generate HTML content for a single investigation."""
  html_content = [
    '<head>',
    f'<title>Investigation {investigation.prompt}</title>',
    '<link href="https://cdn.jsdelivr.net/npm/tailwindcss@^2/dist/tailwind.min.css" rel="stylesheet">',
    '<script src="https://visjs.github.io/vis-network/standalone/umd/vis-network.min.js"></script>',
    '</head>',
    '<body>',
  ]
  html_content.append('<div class="container mx-auto p-4">')
  html_content.append(f'<h1 class="text-3xl font-bold mb-6">{investigation.prompt}</h1>')

  html_content.append('<div class="space-y-4">')

  html_content.append('<div id="network" class="h-96"></div>')

  if investigation.facts:
    html_content.append('<div class="p-4 border rounded-lg shadow bg-white">')
    html_content.append('<p class="font-bold">Facts</p>')
    html_content.append('<ul class="list-disc list-inside">')
    for fact in investigation.facts:
      html_content.append(f'<li class="text-xs text-gray-500">{fact}</li>')
    html_content.append('</ul>')
    html_content.append('</div>')

  for file_name, entry in investigation.data_frames.items():
    html_content.append('<div class="p-4 border rounded-lg shadow bg-white">')
    html_content.append(f'<p class="font-bold">{entry.reason_created}</p>')
    # html_content.append(f'<p class="text-sm text-gray-500 whitespace-pre-wrap">{entry.file_summary}</p>')
    filepath = investigation.dir / entry.filename
    df = pd.read_parquet(filepath)
    html_content.extend(generate_dataframe_html(df))
    html_content.append('</div>')

  for file_id, entry in investigation.files.items():
    html_content.append('<div class="p-4 border rounded-lg shadow bg-white">')
    html_content.append(f'<p class="font-bold">{entry.reason_created}</p>')

    if entry.file_summary:
      html_content.append(f'<p class="text-sm text-gray-500">{entry.file_summary}</p>')

    html_content.append('</div>')

  if investigation.summary:
    html_content.append('<div class="p-4 border rounded-lg shadow bg-white">')
    html_content.append('<p class="font-bold">Summary</p>')
    html_content.append(f'<p class="text-sm text-gray-500 whitespace-pre-wrap">{investigation.summary}</p>')
    html_content.append('</div>')

  html_content.append('</div>')
  html_content.append('</div>')

  network_script = generate_vis_js_network(investigation)
  html_content.append(f'<script>{network_script}</script>')

  return '\n'.join(html_content)


def save_investigation_html(investigation: Investigation, output_path: Path) -> None:
  """Generate and save complete HTML document for an investigation."""
  html_parts = ['<!DOCTYPE html>', '<html lang="en">', '<meta charset="UTF-8">', generate_investigation_html(investigation), '</html>']

  output_html = '\n'.join(html_parts)
  with open(output_path, 'w') as f:
    f.write(output_html)
