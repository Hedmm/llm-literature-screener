"""The review protocol: the question and eligibility criteria, loaded from YAML.

Example protocol file::

    question: Does supervised exercise reduce pain in adults with chronic low back pain?
    population: Adults with non-specific low back pain lasting 12 weeks or longer.
    intervention: Supervised exercise therapy.
    comparator: Usual care, no treatment, waiting list or another conservative treatment.
    outcomes: [Pain intensity, Back-specific disability]
    study_designs: [Randomized controlled trials]
    include:
      - Adults aged 18 or older
    exclude:
      - Acute low back pain (less than 12 weeks)
    languages: [English]
    years: 2000-2024

Only ``question`` is required. The criteria are rendered into the prompt in a fixed order;
a list with one item is written on one line, so a protocol with just a question, one
inclusion and one exclusion text gives exactly the prompt used in the validated case study.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TEXT_FIELDS = ("question", "title", "population", "intervention", "comparator")
LIST_FIELDS = ("outcomes", "study_designs", "include", "exclude", "languages")

ALIASES = {
    "exposure": "intervention",
    "intervention_or_exposure": "intervention",
    "comparison": "comparator",
    "outcome": "outcomes",
    "study_design": "study_designs",
    "designs": "study_designs",
    "inclusion_criteria": "include",
    "exclusion_criteria": "exclude",
    "language": "languages",
}
"""Alternative key names accepted in protocol files."""

_YEAR_RANGE = re.compile(r"^\s*(\d{4})?\s*[-–]\s*(\d{4})?\s*$")


class CriteriaError(ValueError):
    """Raised when a protocol is missing the question or has an invalid value."""


@dataclass(frozen=True)
class YearRange:
    """An inclusive range of publication years; either end may be open."""

    start: int | None = None
    end: int | None = None

    def describe(self) -> str:
        """Describe the range in words, for the prompt."""
        if self.start is not None and self.end is not None:
            return f"{self.start} to {self.end}"
        if self.start is not None:
            return f"{self.start} or later"
        return f"{self.end} or earlier"

    def to_text(self) -> str:
        """Write the range in the ``2000-2024`` form accepted by protocol files."""
        return f"{self.start or ''}-{self.end or ''}"


@dataclass(frozen=True)
class Criteria:
    """The review question and eligibility criteria shown to the model.

    Attributes:
        question: The review question (required).
        title: Optional short name of the review, used in reports.
        population: Population or participants.
        intervention: Intervention or exposure.
        comparator: Comparator or control.
        outcomes: Outcomes of interest.
        study_designs: Eligible study designs.
        include: Other inclusion criteria.
        exclude: Exclusion criteria.
        languages: Eligible publication languages.
        years: Eligible publication years.
    """

    question: str
    title: str = ""
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcomes: tuple[str, ...] = ()
    study_designs: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    years: YearRange | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Criteria:
        """Build criteria from a mapping such as a parsed YAML document.

        Raises:
            CriteriaError: If the question is missing, a key is unknown or a value has the
                wrong type.
        """
        allowed = (*TEXT_FIELDS, *LIST_FIELDS, "years")
        values: dict[str, Any] = {}
        for raw_key, value in data.items():
            key = ALIASES.get(str(raw_key).strip().lower(), str(raw_key).strip().lower())
            if key not in allowed:
                raise CriteriaError(
                    f"Unknown protocol field {raw_key!r}. Allowed fields: {', '.join(allowed)}."
                )
            if key in values:
                raise CriteriaError(f"The protocol sets {key!r} more than once.")
            if key in TEXT_FIELDS:
                values[key] = _text(key, value)
            elif key in LIST_FIELDS:
                values[key] = _items(key, value)
            else:
                values[key] = _years(value)
        if not values.get("question"):
            raise CriteriaError("The protocol needs a 'question' field with the review question.")
        return cls(**values)

    @classmethod
    def from_yaml_text(cls, text: str) -> Criteria:
        """Parse criteria from YAML text.

        Raises:
            CriteriaError: If the YAML is invalid or does not describe a protocol.
        """
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise CriteriaError(f"The protocol is not valid YAML: {error}") from None
        if not isinstance(data, Mapping):
            raise CriteriaError("The protocol must be a YAML mapping such as 'question: ...'.")
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: Path) -> Criteria:
        """Load criteria from a YAML protocol file."""
        return cls.from_yaml_text(path.read_text(encoding="utf-8"))

    def to_dict(self) -> dict[str, Any]:
        """Return the non-empty fields as plain data, for saving next to the results."""
        data: dict[str, Any] = {}
        for name in TEXT_FIELDS:
            value = getattr(self, name)
            if value:
                data[name] = value
        for name in LIST_FIELDS:
            items = getattr(self, name)
            if items:
                data[name] = list(items)
        if self.years is not None:
            data["years"] = self.years.to_text()
        return data

    def to_yaml(self) -> str:
        """Return the protocol as YAML that :meth:`from_yaml_text` reads back."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=100)

    def render(self) -> str:
        """Render the question and criteria as the first block of the screening prompt."""
        lines = [f"Review question: {self.question}"]
        for label, value in (
            ("Population", self.population),
            ("Intervention or exposure", self.intervention),
            ("Comparator", self.comparator),
        ):
            if value:
                lines.append(f"{label}: {value}")
        if self.outcomes:
            lines.append(f"Outcomes: {'; '.join(self.outcomes)}")
        if self.study_designs:
            lines.append(f"Study designs: {'; '.join(self.study_designs)}")
        if self.languages:
            lines.append(f"Languages: {', '.join(self.languages)}")
        if self.years is not None:
            lines.append(f"Publication years: {self.years.describe()}")
        lines.extend(_render_items("Include", self.include))
        lines.extend(_render_items("Exclude", self.exclude))
        return "\n".join(lines)


def _render_items(label: str, items: Sequence[str]) -> list[str]:
    if not items:
        return []
    if len(items) == 1:
        return [f"{label}: {items[0]}"]
    return [f"{label}:", *(f"- {item}" for item in items)]


def _text(key: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return " ".join(str(value).split())
    raise CriteriaError(f"Protocol field {key!r} must be text, not {type(value).__name__}.")


def _items(key: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise CriteriaError(f"Protocol field {key!r} must be text or a list of texts.")
    items = []
    for item in value:
        if isinstance(item, (Mapping, list, tuple)) or item is None:
            raise CriteriaError(f"Each entry of {key!r} must be text.")
        text = " ".join(str(item).split())
        if text:
            items.append(text)
    return tuple(items)


def _year(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        raise CriteriaError(f"'years' must contain four-digit years, not {value!r}.") from None
    if not 1000 <= year <= 2999:
        raise CriteriaError(f"'years' must contain four-digit years, not {value!r}.")
    return year


def _years(value: Any) -> YearRange | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise CriteriaError("'years' must be a range such as 2000-2024.")
    if isinstance(value, int):
        start, end = _year(value), None
    elif isinstance(value, str):
        match = _YEAR_RANGE.match(value)
        if match:
            start, end = _year(match.group(1)), _year(match.group(2))
        else:
            start, end = _year(value.strip()), None
    elif isinstance(value, Mapping):
        unknown = set(value) - {"from", "to", "start", "end"}
        if unknown:
            raise CriteriaError("'years' accepts the keys 'from' and 'to'.")
        start = _year(value.get("from", value.get("start")))
        end = _year(value.get("to", value.get("end")))
    elif isinstance(value, Sequence) and len(value) == 2:
        start, end = _year(value[0]), _year(value[1])
    else:
        raise CriteriaError("'years' must be a range such as 2000-2024.")
    if start is None and end is None:
        return None
    if start is not None and end is not None and start > end:
        raise CriteriaError(f"'years' starts after it ends: {start} > {end}.")
    return YearRange(start, end)
