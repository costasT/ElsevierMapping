import pytest

from sdg_classifier import compile_expression, SDGClassifier


def test_compile_predicate_similarity():
    p = compile_expression('"social protection"')
    # without similarity, plural form should not match exact phrase
    assert not p(None, None, 'social protections')
    # with a high similarity threshold, the plural should match
    assert p(None, None, 'social protections', similarity_threshold=0.9)


def test_classifier_similarity_param():
    clf = SDGClassifier()
    p = compile_expression('"social protection"')
    orig_len = len(clf.predicates)
    clf.predicates.append(p)
    try:
        vec = clf.classify(None, None, 'social protections', similarity_threshold=0.9)
        assert vec[orig_len] == 1
        vec2 = clf.classify(None, None, 'social protections')
        assert vec2[orig_len] == 0
    finally:
        clf.predicates.pop()


def test_classifier_with_fake_embedder():
    # fake embedder: simple deterministic numeric vector per text
    def fake_embed(texts):
        out = []
        for t in texts:
            s = sum(ord(c) for c in (t or "")) % 100
            out.append([float(s), float(len(t or "") % 10)])
        return out

    clf = SDGClassifier(embedder=fake_embed)
    # ensure phrase embeddings were created for each SDG key present
    assert isinstance(clf.phrase_embeddings, dict)
    # call classify with embedding method to exercise code path
    vec = clf.classify("test text", None, None, similarity_threshold=0.1, similarity_method='embed')
    assert isinstance(vec, list) and len(vec) == 17
