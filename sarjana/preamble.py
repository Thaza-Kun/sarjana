from dataclasses import dataclass


@dataclass
class ExecutionOptions:
    Mode: str = "run"
