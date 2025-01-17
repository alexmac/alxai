import hashlib
import inspect
import json
import os
import pickle

CACHE_DIR = ''


def get_cache_dir() -> str:
  global CACHE_DIR
  if not CACHE_DIR:
    _home_dir = os.path.expanduser('~')
    CACHE_DIR = f'{_home_dir}/.open_ai/cache'

    if not os.path.exists(CACHE_DIR):
      os.makedirs(CACHE_DIR)
  return CACHE_DIR


def is_pickleable(x) -> bool:
  try:
    pickle.dumps(x)
    return True
  except (AttributeError, TypeError, pickle.PicklingError):
    return False


def memoize(func):
  async def wrapper(*args, **kwargs):
    source_code = inspect.getsource(func)
    func_key = hashlib.sha256(source_code.encode()).hexdigest()
    arg_keys = hashlib.sha256(json.dumps([str(arg) for arg in args if is_pickleable(arg)] + [str(kwarg) for kwarg in kwargs if is_pickleable(kwarg)]).encode()).hexdigest()

    func_cache_dir = f'{CACHE_DIR}/{func.__name__}/{func_key}'

    cache_file = os.path.join(func_cache_dir, f'{arg_keys}.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as f:
        result = pickle.load(f)
      return result

    result = await func(*args, **kwargs)

    if not os.path.exists(func_cache_dir):
      os.makedirs(func_cache_dir)
    with open(cache_file, 'wb') as f:
      pickle.dump(result, f)

    try:
      cache_file_json = os.path.join(func_cache_dir, f'{arg_keys}.json')
      res = json.dumps(result, indent=2)
      with open(cache_file_json, 'w') as f:
        f.write(res)
    except Exception:
      pass

    return result

  return wrapper
