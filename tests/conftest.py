import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

os.environ["VAX_EXPANSION_CACHE"] = ""  # tests never read or write the on-disk expansion cache
