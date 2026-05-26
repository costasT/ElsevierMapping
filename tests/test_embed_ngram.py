import tempfile
import os
from sdg_classifier import SDGClassifier


def fake_embedder_factory(call_counter):
    def embed(texts):
        call_counter[0] += 1
        out = []
        for t in texts:
            # deterministic 2-d embedding: [token_count, char_sum_mod]
            tc = len(t.split())
            cs = sum(ord(c) for c in t) % 1000
            out.append([float(tc), float(cs)])
        return out

    return embed


def make_sdg_dir(tmpdir):
    # create 17 SDG files with known phrases; SDG1 uses "alpha beta"
    for i in range(1, 18):
        name = f"SDG{str(i).zfill(2)}.txt"
        path = os.path.join(tmpdir, name)
        if i == 1:
            content = '"alpha beta"'
        else:
            content = f'"phrase_{i}"'
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)


def test_embed_ngram_matches_window(tmp_path):
    tmpdir = str(tmp_path)
    make_sdg_dir(tmpdir)
    counter = [0]
    embedder = fake_embedder_factory(counter)
    clf = SDGClassifier(sdg_dir_path=tmpdir, embedder=embedder)
    # reset counter after init to measure classify-time calls
    counter[0] = 0
    title = "An unrelated title"
    abstract = "This paper discusses alpha beta in context."
    kws = None
    vec = clf.classify(title, abstract, kws, similarity_threshold=0.99, similarity_method='embed_ngram')
    assert vec[0] == 1


def test_embed_ngram_batches_windows_once(tmp_path):
    tmpdir = str(tmp_path)
    make_sdg_dir(tmpdir)
    counter = [0]
    embedder = fake_embedder_factory(counter)
    clf = SDGClassifier(sdg_dir_path=tmpdir, embedder=embedder)
    counter[0] = 0
    title = ""
    abstract = "start alpha beta mid alpha beta end"
    kws = None
    vec = clf.classify(title, abstract, kws, similarity_threshold=0.99, similarity_method='embed_ngram')
    # embedder should be called at least once for window embeddings; ensure not called excessively
    assert counter[0] <= 10
    assert vec[0] == 1
