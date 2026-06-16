# Elsevier SDG Mapping — Classifier

This repository provides a compact Python classifier that parses Elsevier-style
search expressions (fielded phrases, AND/OR/NOT, proximity `W/n`, `PRE/n`) and
compiles them into predicates that test whether an article (title, abstract,
keywords) matches any of the 17 SDG mappings.

Quick start

1. Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run the unit tests (recommended):

```bash
python -m pytest -q
```

How to use

	- Returns a list of 17 integers (0/1): position 0 => SDG1, position 16 => SDG17.
	- Any of `title`, `abstract`, `keywords` may be `None`. `keywords` can be a
		string or an iterable of keyword strings.

Example:

```python
from sdg_classifier import classify_sdgs
vec = classify_sdgs("Poverty reduction in rural areas", None, ["poverty"])
print(vec[0])  # 1 if SDG1 matched else 0
```

Optional persistence usage:

```python
from sdg_classifier import SDGClassifier
clf = SDGClassifier(embedding_model_name='all-MiniLM-L6-v2', use_faiss=True)
# Save phrase embeddings and FAISS indices
clf.save_phrase_index('data/phrases.pkl')
clf.save_faiss_indices('data/faiss')

# Later: load into a new classifier (without re-embedding)
clf2 = SDGClassifier(sdg_dir_path='SDG Queries', embedder=None)
clf2.load_phrase_index('data/phrases.pkl')
clf2.load_faiss_indices('data/faiss')
```

Similarity-aware matching

You can enable a word-level similarity fallback when classifying by passing
the optional `similarity_threshold` parameter (float between 0.0 and 1.0) to
the convenience `classify_sdgs(...)` function or to `SDGClassifier.classify()`.
When provided, atoms that do not match exactly are compared using a small
windowed SequenceMatcher-based similarity; this is useful for minor spelling
differences or pluralisation (for example, "social protection" ~ "social
protections"). Example:

```python
from sdg_classifier import classify_sdgs
vec = classify_sdgs(title, abstract, keywords, similarity_threshold=0.9)
```

Similarity methods
------------------

The `similarity_method` parameter controls how embedding-based matching works
after exact expression matching has been attempted. Supported values are:

- `None` (default): exact expression matching only, plus the optional
	`similarity_threshold` word-level fallback described above.
- `"embed"`: whole-document bi-encoder matching. The title, abstract, and
	keywords are combined into a single document embedding and compared with the
	stored SDG phrase embeddings.
- `"embed_ngram"`: sliding n-gram bi-encoder matching. The document is split
	into token-length windows and phrase embeddings are compared against the
	matching-length windows. This is better for finding short phrases inside a
	longer document.

For embedding-based methods, you must provide either `embedding_model_name`
(for example a sentence-transformers model id) or an `embedder` callable that
accepts a list of strings and returns vectors. Example:

```python
from sdg_classifier import SDGClassifier

clf = SDGClassifier(embedding_model_name='all-MiniLM-L6-v2')

vec1 = clf.classify(title, abstract, keywords, similarity_threshold=0.75, similarity_method='embed')
vec2 = clf.classify(title, abstract, keywords, similarity_threshold=0.80, similarity_method='embed_ngram')
```

If the embedding dependencies are not installed, the embedding-based options
will be skipped or disabled, but the exact matcher and similarity threshold
fallback remain available.

Embedding-based similarity
--------------------------

For stronger semantic matching you can supply an embedding model or an
embedder function to `SDGClassifier`. This is optional and requires either
providing `embedding_model_name` (a sentence-transformers model id) or an
`embedder` callable that accepts a list of strings and returns a list of
vector-like sequences. Then call `classify(..., similarity_method='embed', similarity_threshold=0.7)`.

Example (with sentence-transformers installed):

```python
from sdg_classifier import SDGClassifier
clf = SDGClassifier(embedding_model_name='all-MiniLM-L6-v2')
vec = clf.classify(title, abstract, keywords, similarity_threshold=0.7, similarity_method='embed')
```

Sliding n-gram bi-encoder matching
----------------------------------

For more precise local matching (e.g. matching a short phrase inside a longer
abstract) the classifier supports a sliding n-gram bi-encoder approach. Set
`similarity_method='embed_ngram'` when calling `classify()` (or use
`classify_sdgs(..., similarity_method='embed_ngram')` which delegates to the
default classifier). The classifier will create token-length windows across the
document (1..N tokens) and compare phrase embeddings to window embeddings.

Example:

```python
from sdg_classifier import SDGClassifier
clf = SDGClassifier(embedding_model_name='all-MiniLM-L6-v2')
vec = clf.classify(title, abstract, keywords, similarity_threshold=0.8, similarity_method='embed_ngram')
```

FAISS acceleration (optional)
------------------------------

If you need faster matching at scale, you can enable optional FAISS-backed
indices during classifier construction. Install FAISS (CPU build) and set
`use_faiss=True` when creating the classifier. The code will attempt to build
per-SDG, per-phrase-length FAISS indices to speed up nearest-neighbour
searches. If FAISS is not installed the classifier will fall back to a
pure-Python nearest-neighbour search.

Install FAISS (CPU):

```bash
python -m pip install faiss-cpu
```

Then create the classifier with FAISS enabled:

```python
clf = SDGClassifier(embedding_model_name='all-MiniLM-L6-v2', use_faiss=True)
```

Object-oriented usage

The module exposes an `SDGClassifier` class which loads and compiles the
mapping files once. A module-level `DEFAULT_CLASSIFIER` is provided for
backwards compatibility.

`SDGClassifier` parameters

- `sdg_dir_path`: optional path to the directory that contains the SDG mapping
	files. If omitted, the bundled `SDG Queries/` folder is used.
- `embedder`: optional callable that takes a list of strings and returns a list
	of embedding vectors. Use this if you already have an embedding model loaded
	in memory.
- `embedding_model_name`: optional sentence-transformers model id. If provided
	and `embedder` is omitted, the classifier will try to load the model lazily.
- `use_faiss`: enables FAISS-backed nearest-neighbor search for embedding-based
	matching. This is only useful when `embedder` or `embedding_model_name` is
	supplied.
- `cache_dir`: path used to save or load the phrase index with
	`save_phrase_index(...)` and `load_phrase_index(...)`. A `.pkl` file path is
	treated as the index file itself; otherwise it is treated as a cache folder.
- `auto_load_cache`: when `True`, the classifier will try to restore a saved
	phrase index and FAISS indices from `cache_dir` / `cache_faiss_dir` during
	initialization.
- `cache_faiss_dir`: optional directory for FAISS index files when persisting
	or restoring cache data.

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

`classify(...)` parameters

- `title`: article title text. You can pass `None` if the title is unavailable.
- `abstract`: article abstract text. You can pass `None` if the abstract is
	unavailable.
- `keywords`: either a single keyword string or an iterable of keyword
	strings. You can pass `None` if keywords are unavailable.
- `similarity_threshold`: optional float in the range 0.0 to 1.0 used for the
	word-level SequenceMatcher fallback. Higher values are stricter; lower
	values are more permissive.
- `similarity_method`: optional embedding mode selector:
	- `None`: exact expression matching plus the optional threshold fallback.
	- `"embed"`: whole-document embedding comparison.
	- `"embed_ngram"`: sliding n-gram embedding comparison.

The method returns a 17-element list of 0/1 flags, one per SDG.

Data and mappings

- Mapping files are stored in the `SDG Queries/` directory (SDG01.txt .. SDG17.txt).
- During import the module compiles each SDG expression into a predicate. If
	compilation fails for a mapping the loader previously fell back to a simple
	quoted-phrase matcher; recent tokenizer/parser improvements aim to avoid
	those fallbacks and fully compile the original expressions.

Tests and development

- Tests live in the `tests/` folder and are run with `pytest`.
- A small set of parser-feature unit tests exercise proximity, field-scoped
	atoms, wildcard handling, and nested `FIELD(...)` inner expressions.
- Install any additional development tools you prefer locally; only runtime and
	test dependencies are listed in `requirements.txt`.

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
