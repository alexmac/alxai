# Usage

# Setup your virtual env

- Install uv (https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
- run "uv sync"

- brew install ansifilter

# Setup OpenAI API key

- Create a file in ~/.open_ai/config.json with the following content:

```json
{
  "orgs": {
    "your_org_id": {
      "secret": "your_api_key"
    }
  }
}
```
