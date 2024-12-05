from pydantic import BaseModel
from typing import List

class PredictionList(BaseModel):
    inp: List[float]  # A list of integers, 1 for attack and 0 for non-attack
