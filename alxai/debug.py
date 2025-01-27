import os
from datetime import datetime

BASE_OUTPUT_DIR = 'output/conversations'
CURRENT_RUN_DIR = os.path.join(BASE_OUTPUT_DIR, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(CURRENT_RUN_DIR, exist_ok=True)
