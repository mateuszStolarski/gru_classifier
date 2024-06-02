from dataclasses import dataclass


@dataclass
class WrapperConfig:
    num_warmup_steps: int
    max_steps: int
    num_cycles: int
