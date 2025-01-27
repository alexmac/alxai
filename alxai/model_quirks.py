def strip_code_prefix(txt: str) -> str:
  if txt.startswith('```json'):
    txt = txt[7:-3]
  txt = txt.strip()
  return txt
