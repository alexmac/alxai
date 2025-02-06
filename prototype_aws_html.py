from pathlib import Path

from investigation.investigation import Investigation
from investigation.summarize_as_html import save_investigation_html


def main():
  # Base output directory
  base_dir = Path('output/investigations')

  # Process each investigation directory
  for investigation_dir in base_dir.iterdir():
    if investigation_dir.is_dir():
      # Read master index and parse with Pydantic
      with open(investigation_dir / 'master_index.json', 'r') as f:
        investigation = Investigation.model_validate_json(f.read(), strict=False)
        investigation.dir = investigation_dir

      # Generate and save HTML
      output_path = investigation_dir / 'index.html'
      save_investigation_html(investigation, output_path)


if __name__ == '__main__':
  main()
