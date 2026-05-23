import pytest


@pytest.mark.parametrize("sdg_num", list(range(1, 18)))
def test_sdg_has_matching_sample(sdg_num, classifier):
    phrases = classifier.phrases.get(sdg_num, [])
    if not phrases:
        pytest.skip(f"No extracted phrases for SDG{sdg_num}")
    idx = sdg_num - 1
    matched = False
    last_vec = None
    # try single phrases first
    for sample in phrases[:10]:
        vec = classifier.classify(sample, None, sample)
        last_vec = vec
        if 0 <= idx < len(vec) and vec[idx] == 1:
            matched = True
            break
    # try simple combinations if single phrases didn't match
    if not matched:
        pool = phrases[:200]
        tried = 0
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                a = pool[i]
                b = pool[j]
                vec = classifier.classify(a, None, b)
                last_vec = vec
                tried += 1
                if 0 <= idx < len(vec) and vec[idx] == 1:
                    matched = True
                    break
                vec = classifier.classify(a, b, None)
                last_vec = vec
                tried += 1
                if 0 <= idx < len(vec) and vec[idx] == 1:
                    matched = True
                    break
                if tried > 2000:
                    break
            if matched or tried > 2000:
                break
    assert matched, f"SDG{sdg_num} none of sample phrases matched; last_vec={last_vec}"
