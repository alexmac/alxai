from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel

from investigation.investigation import Investigation


class FileEntry(BaseModel):
  filename: str
  file_type: str
  reason_created: str
  file_schema: Optional[str] = None
  file_summary: Optional[str] = None


class MasterIndex(BaseModel):
  prompt: str
  files: Dict[str, FileEntry]


def generate_html_for_investigation(investigation_dir):
  """Generate HTML content for a single investigation directory."""

  # Read master index and parse with Pydantic
  with open(investigation_dir / 'master_index.json', 'r') as f:
    master_index = MasterIndex.model_validate_json(f.read())

  investigation = Investigation(master_index.prompt, investigation_dir)

  html_content = [
    '<head>',
    f'<title>Investigation {investigation.prompt}</title>',
    '<link href="https://cdn.jsdelivr.net/npm/tailwindcss@^2/dist/tailwind.min.css" rel="stylesheet">',
    '</head>',
    '<body>',
  ]
  html_content.append('<div class="container mx-auto p-4">')
  html_content.append(f'<h1 class="text-3xl font-bold mb-6">{investigation.prompt}</h1>')

  html_content.append('<div class="space-y-4">')

  for file_id, entry in master_index.files.items():
    html_content.append('<div class="p-4 border rounded-lg shadow bg-white">')
    html_content.append(f'<p class="font-bold">{entry.reason_created}</p>')

    if entry.file_summary:
      html_content.append(f'<p class="text-sm text-gray-500">{entry.file_summary}</p>')

    html_content.append('</div>')

  html_content.append('</div>')
  html_content.append('</div>')

  return '\n'.join(html_content)


def main():
  # Base output directory
  base_dir = Path('output/investigations')

  # Process each investigation directory
  for investigation_dir in base_dir.iterdir():
    if investigation_dir.is_dir():
      # Start HTML document for each investigation
      html_parts = [
        '<!DOCTYPE html>',
        '<html>',
      ]
      html_parts.append(generate_html_for_investigation(investigation_dir))
      html_parts.extend(['</html>'])

      # Write the HTML file for the current investigation
      output_html = '\n'.join(html_parts)
      output_path = investigation_dir / 'index.html'
      with open(output_path, 'w') as f:
        f.write(output_html)


if __name__ == '__main__':
  main()
