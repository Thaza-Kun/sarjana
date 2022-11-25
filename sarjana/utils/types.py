from dataclasses import dataclass
from datetime import datetime
from typing import List, TypeVar, Tuple, Callable, Optional

from pandas import DataFrame


@dataclass
class ParameterRecord:
    columns: List[str]
    min_cluster_size: int
    seed: int


@dataclass
class ScoreRecord:
    value: float
    metric: str


@dataclass
class WorkflowMetadata:
    timestamp: datetime
    parameters: ParameterRecord
    score: ScoreRecord
    name: str = ""


WorkflowResult = TypeVar("WorkflowResult", bound=Tuple[DataFrame, WorkflowMetadata])
DataFunc = TypeVar("DataFunc", bound=Callable[..., DataFrame])
FlowFunc = TypeVar("FlowFunc", bound=Callable[..., WorkflowResult])
StringOptions = TypeVar("StringOptions", bound=Optional[List[str]])
