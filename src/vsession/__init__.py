"""V-Session experiment and evaluation utilities."""

from .data import Example, load_examples
from .evaluation import EvaluationResult, evaluate_response
from .prompts import PromptTemplate, load_prompt
from .rsq import DIMENSIONS, WEIGHT_PRESETS, aggregate_rsq
from .structure import StructureAudit, audit_vsession_structure

__all__ = [
    "EvaluationResult",
    "Example",
    "DIMENSIONS",
    "PromptTemplate",
    "StructureAudit",
    "WEIGHT_PRESETS",
    "aggregate_rsq",
    "audit_vsession_structure",
    "evaluate_response",
    "load_examples",
    "load_prompt",
]

__version__ = "0.2.0"
