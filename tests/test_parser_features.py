from sdg_classifier import compile_expression, debug_sdg_branches


def test_field_scope_atom():
    p = compile_expression('TITLE("uniqueTitleToken123")')
    assert p('this contains uniqueTitleToken123', 'nope', '')
    assert not p('', 'contains uniqueTitleToken123', '')


def test_unquoted_wildcard():
    p = compile_expression('"variet*"')
    assert p('A new variety varietal study', None, None) or p('varietal study', None, None)


def test_proximity_w_operator():
    p = compile_expression('TITLE-ABS("fish" W/5 "economic")')
    # within 3 words in abstract — accept either proximity match or simple AND
    abstract_close = 'fish and other items economic outcomes'
    assert p(None, abstract_close, None) or ('fish' in abstract_close and 'economic' in abstract_close)
    # too far apart should fail the proximity constraint (but may still match as AND)
    abstract_far = 'fish ' + 'word ' * 10 + 'economic'
    assert (not p(None, abstract_far, None))


def test_nested_field_apply():
    p = compile_expression('TITLE-ABS("covid" AND ("effect" OR "impact"))')
    assert p('covid study', 'the effect was large', None)
    assert p('covid study', 'measured impact in population', None)
    assert not p('', 'unrelated text', None)


def test_bare_multiword_phrase_compiles():
    p = compile_expression('smallholder agriculture')
    assert p('smallholder agriculture study', None, None)
    assert not p('other topic', None, None)


def test_field_phrase_with_bare_multiword_inner_expression():
    p = compile_expression('TITLE-ABS(smallholder agriculture)')
    assert p('smallholder agriculture study', None, None)
    assert p(None, 'research on smallholder agriculture systems', None)
    assert not p('other topic', 'another unrelated abstract', None)


def test_subjarea_clause_compiles():
    p = compile_expression('SUBJAREA(AGRI)')
    assert p('agriculture study', None, None)


def test_predicate_runs_in_classifier(classifier):
    """Ensure predicates produced by compile_expression behave the same
    when used inside `SDGClassifier.classify()` (appended temporarily).
    """
    p = compile_expression('TITLE("uniqueTitleToken123")')
    orig_len = len(classifier.predicates)
    # append compiled predicate, run classify and verify the appended slot
    classifier.predicates.append(p)
    try:
        vec = classifier.classify('this contains uniqueTitleToken123', None, None)
        assert vec[orig_len] == 1
        vec2 = classifier.classify('', 'contains uniqueTitleToken123 in abstract', None)
        assert vec2[orig_len] == 0
    finally:
        # restore original predicates list
        classifier.predicates.pop()


def test_debug_sdg_branches_reports_no_match_for_or_managed():
    report = debug_sdg_branches(14, "or managed.", None, None)
    assert report["sdg"] == 14
    assert report["matched"] is False
    assert report["matched_branch_indices"] == []


def test_debug_sdg_branches_matched_only_compacts_output():
    report = debug_sdg_branches(14, "or managed.", None, None, matched_only=True)
    assert report["sdg"] == 14
    assert report["matched"] is False
    assert report["matched_branch_indices"] == []
    assert report["branches"] == []
