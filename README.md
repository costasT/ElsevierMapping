# Elsevier SDG Mapping — Classifier

This repository provides a compact Python classifier that parses Elsevier-style
search expressions (fielded phrases, AND/OR/NOT, proximity `W/n`, `PRE/n`) and
compiles them into predicates that test whether an article (title, abstract,
keywords) matches any of the 17 SDG mappings.

Quick start

1. Install test / formatting tools (optional):

```bash
python -m pip install pytest black
```

2. Run the unit tests (recommended):

```bash
python -m pytest -q
```

How to use

- The primary API is `classify_sdgs(title, abstract, keywords) -> List[int]`.
	- Returns a list of 17 integers (0/1): position 0 => SDG1, position 16 => SDG17.
	- Any of `title`, `abstract`, `keywords` may be `None`. `keywords` can be a
		string or an iterable of keyword strings.

Example:

```python
from sdg_classifier import classify_sdgs
vec = classify_sdgs("Poverty reduction in rural areas", None, ["poverty"])
print(vec[0])  # 1 if SDG1 matched else 0
```

Object-oriented usage

The module now exposes an `SDGClassifier` class which loads and compiles the
mapping files once. A module-level `DEFAULT_CLASSIFIER` is provided for
backwards compatibility.

```python
from sdg_classifier import SDGClassifier, DEFAULT_CLASSIFIER

# Create your own classifier (optional, uses default mapping directory)
clf = SDGClassifier()
vec = clf.classify(title, abstract, keywords)

# Or use the shared/default instance
vec2 = DEFAULT_CLASSIFIER.classify(title, abstract, keywords)
```

The old convenience function `classify_sdgs(...)` remains and delegates to the
default classifier so existing code keeps working.

Data and mappings

- Mapping files are stored in the `SDG 2023 Queries/` directory (SDG01.txt .. SDG17.txt).
- During import the module compiles each SDG expression into a predicate. If
	compilation fails for a mapping the loader previously fell back to a simple
	quoted-phrase matcher; recent tokenizer/parser improvements aim to avoid
	those fallbacks and fully compile the original expressions.

Tests and development

- Tests live in the `tests/` folder and are run with `pytest`.
- A small set of parser-feature unit tests exercise proximity, field-scoped
	atoms, wildcard handling, and nested `FIELD(...)` inner expressions.
- Use `black sdg_classifier.py` to format the main module.

CLI

The module can also be used as a small CLI. Example invocations:

```bash
# demo example
python -m sdg_classifier --demo

# classify a single article
python -m sdg_classifier -t "Title text" -a "Abstract text" -k keyword1 -k keyword2

# list SDG phrase counts
python -m sdg_classifier --list-sdgs

# check which SDGs used a fallback at import
python -m sdg_classifier --check-fallbacks
```

Cleanup notes

- Temporary debug helpers used during development have been removed. The test
	suite now exercises the mapping compilation and predicate behavior for all
	17 SDGs.

Contact / next steps

- If you want further improvements, common requests are: expanding proximity
	semantics, supporting additional field names, or exporting compiled predicates
	for faster use in production systems.
