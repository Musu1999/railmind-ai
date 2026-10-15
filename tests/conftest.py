"""Test setup: isolated artifact directory, models trained once per session."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_TMP = tempfile.mkdtemp(prefix="railmind-test-")
os.environ["RAILMIND_ARTIFACT_DIR"] = _TMP
os.environ["RAILMIND_DB"] = str(Path(_TMP) / "test.db")
os.environ.pop("RAILMIND_LLM", None)
sys.path.insert(0, str(ROOT))

subprocess.run([sys.executable, str(ROOT / "scripts" / "train_all.py")], check=True, env=os.environ.copy(),
               stdout=subprocess.DEVNULL)
