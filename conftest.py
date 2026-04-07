"""Pytest configuration — ensure project root is on PYTHONPATH."""
import sys
from pathlib import Path

# Add project root to path so tests can import server.*
sys.path.insert(0, str(Path(__file__).parent))
