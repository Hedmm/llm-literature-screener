"""Pre-screen titles and abstracts for systematic reviews with an open LLM.

The model reads the review question, the eligibility criteria and each record's title and
abstract, and answers Yes or No. The score is the logit difference between the two answers,
and a threshold calibrated on records the team has already screened keeps a target share of
the known inclusions. People make the final decisions.
"""

from llm_literature_screener.record import Assessment, Decision, Record, sigmoid

__version__ = "0.1.0"

__all__ = ["Assessment", "Decision", "Record", "__version__", "sigmoid"]
