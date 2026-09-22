from typing import Literal
from pydantic import BaseModel

PuzzleType = Literal["sudoku", "logic_grid", "skyscraper", "abstract_math"]
ReasoningLength = Literal["minimal", "natural", "extended"]

class GeneratedPuzzle(BaseModel):
    puzzle_type: str
    puzzle: str


class IsBenignPuzzle(BaseModel):
    is_benign_puzzle: bool
