from typing import Any, Awaitable, Callable, Dict, List

import pandas as pd


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
  """Recursively flatten a nested dictionary."""
  items: List[tuple] = []
  for k, v in d.items():
    new_key = f'{parent_key}{sep}{k}' if parent_key else k

    if isinstance(v, dict):
      items.extend(flatten_dict(v, new_key, sep=sep).items())
    elif not isinstance(v, list):
      items.append((new_key, v))

  return dict(items)


async def json_to_dataframes(json_data: List[Dict], name: str, get_id_key: Callable[[Dict], Awaitable[str]]) -> Dict[str, pd.DataFrame]:
  # Track array fields that need separate files
  array_fields = {}
  flattened_records = []

  if len(json_data) == 0:
    return {}

  id_key = await get_id_key(json_data[0])

  # First pass - identify array fields and flatten records
  for record in json_data:
    flat_record = {}

    # Ensure ID field exists
    if id_key not in record:
      raise ValueError(f"ID key '{id_key}' not found in record")

    record_id = record[id_key]

    # Process each field
    for key, value in record.items():
      if isinstance(value, list):
        if key not in array_fields:
          array_fields[key] = []
        # Store array data with parent ID
        for item in value:
          if isinstance(item, dict):
            item[id_key] = record_id
            array_fields[key].append(item)
          else:
            array_fields[key].append({id_key: record_id, 'value': item})
      elif isinstance(value, dict):
        # Flatten nested dictionaries
        nested_flat = flatten_dict(value, parent_key=key)
        flat_record.update(nested_flat)
      else:
        flat_record[key] = value

    flattened_records.append(flat_record)

  dfs = {}
  main_df = pd.DataFrame(flattened_records)
  dfs[name] = main_df

  for field_name, array_data in array_fields.items():
    if array_data:
      array_df = pd.DataFrame(array_data)
      dfs[field_name] = array_df

  return dfs


async def extract_dataframes_from_json(obj: Dict | List, name: str, get_id_key: Callable[[Dict], Awaitable[str]]) -> Dict[str, pd.DataFrame]:
  if isinstance(obj, dict):
    keys = list(obj.keys())
    if len(keys) == 1:
      return await extract_dataframes_from_json(obj[keys[0]], keys[0], get_id_key)
    else:
      assert False, f'Expected a single key in dictionary, got {keys}'
  elif isinstance(obj, list):
    return await json_to_dataframes(obj, name, get_id_key)
  else:
    assert False, f'Expected a dictionary or list, got {type(obj)}'
