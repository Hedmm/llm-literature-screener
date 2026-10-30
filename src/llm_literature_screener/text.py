"""Small helpers for the wording of reports and messages."""


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """Return ``count`` followed by the singular or plural form of a noun.

    >>> plural(1, "record"), plural(3, "record"), plural(2, "study", "studies")
    ('1 record', '3 records', '2 studies')
    """
    word = singular if count == 1 else (plural_form or singular + "s")
    return f"{count:,} {word}"
