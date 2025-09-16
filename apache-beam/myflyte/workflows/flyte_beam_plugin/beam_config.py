# beam_config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class BeamJobConfig:
    runner: str = "DirectRunner"  # Could be DataflowRunner, FlinkRunner in the future
    temp_location: Optional[str] = None
    project: Optional[str] = None
    region: Optional[str] = None
