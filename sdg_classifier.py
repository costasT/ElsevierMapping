import re
import os
from difflib import SequenceMatcher
from typing import Iterable, Union, List, Dict, Callable, Optional, Any
import pickle
import pathlib
import json

# --- Expression-based mapping support ---------------------------------
# Basic grammar supports atoms of the form FIELD1-FIELD2("phrase") or
# "phrase" (quoted) and boolean operators AND, OR and parentheses. Fields
# can be TITLE, ABS, KW. Example: ("poverty" OR TITLE-ABS("unesco")) AND KW("social protection")

_pattern_cache: dict = {}

_FIELD_LIST_PATTERN = re.compile(
    r"\s*((?:TITLE|ABS|KW|AUTHKEY|KEY)(?:-(?:TITLE|ABS|KW|AUTHKEY|KEY))*)\s*\(",
    flags=re.IGNORECASE,
)
_SINGLE_QUOTED_LITERAL_PATTERN = re.compile(
    r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''
)


def _normalize_fields(fields_raw: str) -> List[str]:
    """Normalize parsed field specifiers to internal field names."""
    fields = []
    for f in fields_raw.upper().split("-"):
        if f in ("AUTHKEY", "KEY"):
            fields.append("KW")
        else:
            fields.append(f)
    return fields


def _unwrap_single_quoted_literal(text: str) -> str:
    """Strip wrapping quotes only when text is a single quoted literal."""
    if _SINGLE_QUOTED_LITERAL_PATTERN.fullmatch(text):
        return text[1:-1]
    return text


def _normalize_similarity_text(value: Optional[str]) -> str:
    text = (value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_similarity_match(phrase: str, text: str, threshold: float) -> bool:
    phrase_norm = _normalize_similarity_text(phrase)
    text_norm = _normalize_similarity_text(text)
    if not phrase_norm or not text_norm:
        return False

    phrase_tokens = phrase_norm.split()
    text_tokens = text_norm.split()
    if not phrase_tokens or not text_tokens:
        return False

    target = " ".join(phrase_tokens)
    window_size = len(phrase_tokens)

    if window_size == 1:
        return any(
            SequenceMatcher(None, target, token).ratio() >= threshold
            for token in text_tokens
        )

    if len(text_tokens) < window_size:
        return SequenceMatcher(None, target, text_norm).ratio() >= threshold

    for start in range(len(text_tokens) - window_size + 1):
        window = " ".join(text_tokens[start : start + window_size])
        if SequenceMatcher(None, target, window).ratio() >= threshold:
            return True
    return False


def _call_predicate(
    pred: Callable,
    title: Optional[str],
    abstract: Optional[str],
    keywords: Optional[Union[str, Iterable[str]]],
    similarity_threshold: Optional[float] = None,
) -> bool:
    if similarity_threshold is None:
        return bool(pred(title, abstract, keywords))

    try:
        return bool(pred(title, abstract, keywords, similarity_threshold))
    except TypeError:
        return bool(pred(title, abstract, keywords))


def _get_pattern(phrase: str) -> re.Pattern:
    p = _pattern_cache.get(phrase)
    if p is None:
        # compile phrase using Elsevier-style wildcard semantics:
        #  - '*' => '.*'
        #  - spaces => match one or more whitespace (\s+)
        #  - escape other regex metacharacters
        def _elsevier_to_regex(s: str) -> str:
            s = s.strip()
            # collapse multiple spaces
            s = re.sub(r"\s+", " ", s)
            # placeholder for wildcard so we can escape safely
            placeholder = "__WILDCARD__"
            s = s.replace("*", placeholder)
            # escape regex metacharacters
            s = re.escape(s)
            # turn escaped spaces into \s+ to be flexible about whitespace
            s = s.replace(r"\ ", r"\s+")
            # restore wildcard regex but restrict it so it doesn't span
            # across whitespace (i.e. '*' matches within a token only)
            s = s.replace(placeholder, r"[^\s]*")
            # anchor to word boundaries where practical
            return r"\b" + s + r"\b"

        pattern = _elsevier_to_regex(phrase)
        p = re.compile(pattern, flags=re.IGNORECASE)
        _pattern_cache[phrase] = p
    return p


def _match_in_field(
    phrase: str, field: str, title: str, abstract: str, kws: str
) -> bool:
    pat = _get_pattern(phrase)
    # normalize field tokens: accept AUTHKEY / KEY as keyword fields
    f = field.upper()
    if f in ("AUTHKEY", "KEY"):
        f = "KW"
    if f == "TITLE":
        return bool(pat.search(title or ""))
    if f == "ABS":
        return bool(pat.search(abstract or ""))
    if f == "KW":
        return bool(pat.search(kws or ""))
    # Unknown field -> do not match
    return False


def _combined_article_text(
    title: Optional[str],
    abstract: Optional[str],
    keywords: Optional[Union[str, Iterable[str]]],
) -> str:
    parts = [title or "", abstract or ""]
    if keywords is None:
        pass
    elif isinstance(keywords, str):
        parts.append(keywords)
    else:
        parts.append(" ".join(str(item) for item in keywords if item))
    return " ".join(part for part in parts if part).strip()


def _keywords_to_text(keywords: Optional[Union[str, Iterable[str]]]) -> str:
    """Return keyword payload as a single text string."""
    if keywords is None:
        return ""
    if isinstance(keywords, str):
        return keywords
    try:
        return " ".join(keywords)
    except Exception:
        try:
            return str(keywords)
        except Exception:
            return ""


def _build_doc_text(
    title: Optional[str],
    abstract: Optional[str],
    keywords: Optional[Union[str, Iterable[str]]],
) -> str:
    """Compose document text for embedding-based matching."""
    return " ".join(filter(None, [title or "", abstract or "", _keywords_to_text(keywords)]))


def _cosine_similarity_fn():
    """Return a cosine-similarity function using numpy when available."""
    try:
        import numpy as _np

        def _cos(a, b):
            a = _np.asarray(a, dtype=float)
            b = _np.asarray(b, dtype=float)
            denom = _np.linalg.norm(a) * _np.linalg.norm(b)
            return float(_np.dot(a, b) / denom) if denom else 0.0

    except Exception:

        def _cos(a, b):
            sa = sum(x * x for x in a) ** 0.5
            sb = sum(x * x for x in b) ** 0.5
            if sa == 0 or sb == 0:
                return 0.0
            return sum(x * y for x, y in zip(a, b)) / (sa * sb)

    return _cos


def _phrase_present(text: str, phrase: str) -> bool:
    pat = _get_pattern(phrase)
    return bool(pat.search(text or ""))


def _phrase_matches_text(
    phrase: str, text: str, similarity_threshold: Optional[float] = None
) -> bool:
    if _phrase_present(text, phrase):
        return True
    if similarity_threshold is None:
        return False
    return _word_similarity_match(phrase, text, similarity_threshold)


class _Node:
    def eval(self, title, abstract, kws, similarity_threshold=None):
        raise NotImplementedError()


class _Atom(_Node):
    def __init__(self, phrase: str, fields: Union[None, list] = None):
        # fields==None means match any field (title|abstract|keywords)
        self.phrase = phrase
        self.fields = fields

    def eval(self, title, abstract, kws, similarity_threshold=None):
        if not self.fields:
            # any field
            pat = _get_pattern(self.phrase)
            if bool(
                pat.search(title or "")
                or pat.search(abstract or "")
                or pat.search(kws or "")
            ):
                return True
            if similarity_threshold is None:
                return False
            return any(
                _word_similarity_match(self.phrase, text, similarity_threshold)
                for text in (title, abstract, kws)
                if text
            )
        for f in self.fields:
            if _match_in_field(self.phrase, f, title, abstract, kws):
                return True
        if similarity_threshold is not None:
            field_texts = []
            for f in self.fields:
                if f == "TITLE":
                    field_texts.append(title)
                elif f == "ABS":
                    field_texts.append(abstract)
                elif f == "KW":
                    field_texts.append(kws)
            return any(
                _word_similarity_match(self.phrase, text, similarity_threshold)
                for text in field_texts
                if text
            )
        return False


class _And(_Node):
    def __init__(self, left: _Node, right: _Node):
        self.left = left
        self.right = right

    def eval(self, title, abstract, kws, similarity_threshold=None):
        return self.left.eval(
            title, abstract, kws, similarity_threshold=similarity_threshold
        ) and self.right.eval(
            title, abstract, kws, similarity_threshold=similarity_threshold
        )


class _Or(_Node):
    def __init__(self, left: _Node, right: _Node):
        self.left = left
        self.right = right

    def eval(self, title, abstract, kws, similarity_threshold=None):
        return self.left.eval(
            title, abstract, kws, similarity_threshold=similarity_threshold
        ) or self.right.eval(
            title, abstract, kws, similarity_threshold=similarity_threshold
        )


class _AndList(_Node):
    def __init__(self, children):
        self.children = children

    def eval(self, title, abstract, kws, similarity_threshold=None):
        for c in self.children:
            if not c.eval(
                title, abstract, kws, similarity_threshold=similarity_threshold
            ):
                return False
        return True


class _OrList(_Node):
    def __init__(self, children):
        self.children = children

    def eval(self, title, abstract, kws, similarity_threshold=None):
        for c in self.children:
            if c.eval(title, abstract, kws, similarity_threshold=similarity_threshold):
                return True
        return False


class _Not(_Node):
    def __init__(self, child: _Node):
        self.child = child

    def eval(self, title, abstract, kws, similarity_threshold=None):
        return not self.child.eval(
            title, abstract, kws, similarity_threshold=similarity_threshold
        )


class _Prox(_Node):
    def __init__(
        self,
        op: str,
        n: int,
        left: _Node = None,
        right: _Node = None,
        child: _Node = None,
    ):
        self.op = op  # e.g., 'W' or 'PRE'
        self.n = n
        self.left = left
        self.right = right
        self.child = child

    def eval(self, title, abstract, kws, similarity_threshold=None):
        # Field-aware proximity evaluation.
        def field_text(f: str) -> str:
            if f == "TITLE":
                return title or ""
            if f == "ABS":
                return abstract or ""
            if f == "KW":
                return kws or ""
            return ""

        # Helper to find word-based distances between two patterns in a text
        def patterns_within_distance(patL, patR, txt, max_words, directional=False):
            starts = [m.start() for m in patL.finditer(txt)]
            ends = [m.start() for m in patR.finditer(txt)]
            if not starts or not ends:
                return False
            for s in starts:
                for e in ends:
                    if directional and s >= e:
                        continue
                    a, b = (s, e) if s <= e else (e, s)
                    window = txt[a:b]
                    word_count = len(re.findall(r"\w+", window))
                    if word_count <= max_words:
                        return True
            return False

        if self.left is not None and self.right is not None:
            L = self.left
            R = self.right
            # If both sides are atoms, try to evaluate within shared fields.
            if isinstance(L, _Atom) and isinstance(R, _Atom):
                # determine fields to consider
                fields_L = L.fields or ["TITLE", "ABS", "KW"]
                fields_R = R.fields or ["TITLE", "ABS", "KW"]
                # normalize AUTHKEY/KEY to KW already done during parsing
                # try same-field proximity first
                for f in set(fields_L) & set(fields_R):
                    txt = field_text(f)
                    if not txt:
                        continue
                    patL = _get_pattern(L.phrase)
                    patR = _get_pattern(R.phrase)
                    directional = self.op.startswith("PRE")
                    if self.op.startswith("W") or self.op.startswith("NEAR"):
                        if patterns_within_distance(
                            patL, patR, txt, self.n, directional=False
                        ):
                            return True
                    else:
                        # default to unordered proximity
                        if patterns_within_distance(
                            patL, patR, txt, self.n, directional=directional
                        ):
                            return True
                # fallback: check across combined text
                combined = " ".join(
                    filter(None, [title or "", abstract or "", kws or ""])
                )
                patL = _get_pattern(L.phrase)
                patR = _get_pattern(R.phrase)
                directional = self.op.startswith("PRE")

                
                if patterns_within_distance(
                    patL, patR, combined, self.n, directional=directional
                ):
                    return True
                # For atom-vs-atom proximity, do not degrade to logical AND.
                # If terms are present but too far apart, this must remain False.
                return False
            # for non-atom children, evaluate child if present
            try:
                return self.left.eval(
                    title, abstract, kws, similarity_threshold=similarity_threshold
                ) and self.right.eval(
                    title, abstract, kws, similarity_threshold=similarity_threshold
                )
            except Exception:
                return False
        if self.child is not None:
            return self.child.eval(
                title, abstract, kws, similarity_threshold=similarity_threshold
            )
        return False


def _make_predicate_from_ast(root: _Node) -> Callable:
    """Build a callable predicate from a compiled AST root."""

    def predicate(title, abstract, keywords, similarity_threshold=None):
        if keywords is None:
            kws = ""
        elif isinstance(keywords, str):
            kws = keywords
        else:
            kws = " ".join(keywords)
        try:
            return root.eval(
                title or "",
                abstract or "",
                kws,
                similarity_threshold=similarity_threshold,
            )
        except RecursionError:
            return False

    return predicate


def compile_expression(expr: str, return_ast: bool = False):
    """Compile a mapping expression string into a predicate function.

    Supported atoms:
      - FIELD1-FIELD2("phrase")  where FIELD is TITLE, ABS, KW
      - "phrase" (quoted) meaning search any field

    Operators: AND, OR (case-insensitive). Parentheses supported.
    """
    tokens = []
    pos = 0
    length = len(expr)

    def _match_op_at(p):
        m = re.match(r"\s*(AND|OR|NOT)\b", expr[p:], flags=re.IGNORECASE)
        if m:
            return ("OP", m.group(1).upper(), p + m.end())
        return None

    def _match_prox_at(p):
        m = re.match(r"\s*([A-Z]+/[0-9]+)", expr[p:], flags=re.IGNORECASE)
        if m:
            return ("PROX", m.group(1).upper(), p + m.end())
        return None

    def _match_paren_at(p):
        m = re.match(r"\s*([()])", expr[p:])
        if m:
            return (
                ("LPAREN" if m.group(1) == "(" else "RPAREN"),
                m.group(1),
                p + m.end(),
            )
        return None

    def _match_quoted_at(p):
        m = re.match(r"\s*(\"(?:[^\\\"]|\\.)*\"|'(?:[^\\']|\\.)*')", expr[p:])
        if m:
            return ("PHRASE", m.group(1), p + m.end())
        return None

    def _find_matching_paren(start_idx):
        # start_idx points to the '(' char
        i = start_idx
        depth = 0
        in_quote = None
        while i < length:
            ch = expr[i]
            if in_quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_quote:
                    in_quote = None
                i += 1
                continue
            if ch in ('"', "'"):
                in_quote = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1

    while pos < length:
        # skip whitespace
        if expr[pos].isspace():
            pos += 1
            continue
        # operators
        m = _match_op_at(pos)
        if m:
            tokens.append((m[0], m[1]))
            pos = m[2]
            continue
        # proximity
        m = _match_prox_at(pos)
        if m:
            tokens.append((m[0], m[1]))
            pos = m[2]
            continue
        # parens
        m = _match_paren_at(pos)
        if m:
            tokens.append((m[0], m[1]))
            pos = m[2]
            continue
        # field list like TITLE-ABS(
        m = _FIELD_LIST_PATTERN.match(expr[pos:])
        if m:
            fields = _normalize_fields(m.group(1))
            # find the absolute position of the opening paren
            paren_pos = pos + m.end() - 1
            end_pos = _find_matching_paren(paren_pos)
            if end_pos == -1:
                raise ValueError("Unmatched parenthesis in FIELD(...) clause")
            inner = expr[paren_pos + 1 : end_pos].strip()
            # Strip wrapping quotes only when the FIELD(...) content is a
            # single quoted literal, not when it is a nested expression like
            # "A" OR "B".
            phrase = _unwrap_single_quoted_literal(inner)
            tokens.append(("FIELD_PHRASE", (fields, phrase)))
            pos = end_pos + 1
            continue
        # quoted phrase
        m = _match_quoted_at(pos)
        if m:
            tokens.append((m[0], m[1]))
            pos = m[2]
            continue
        # unquoted atom
        m = re.match(r"\s*([^\s()]+)", expr[pos:])
        if m:
            tokens.append(("PHRASE", m.group(1)))
            pos += m.end()
            continue
        raise ValueError(f"Unable to tokenize at: {expr[pos:pos+20]!r}")

    # Parser (recursive descent)
    idx = 0

    def peek():
        return tokens[idx] if idx < len(tokens) else None

    def consume(expected_type=None):
        nonlocal idx
        if idx >= len(tokens):
            return None
        t = tokens[idx]
        idx += 1
        if expected_type and t[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {t}")
        return t

    def parse_atom():
        t = peek()
        if t is None:
            raise ValueError("Unexpected end of expression")
        # unary NOT support
        if t[0] == "OP" and t[1] == "NOT":
            consume("OP")
            child = parse_atom()
            return _Not(child)
        # prefix proximity operator: PROX <atom>
        if t[0] == "PROX":
            tok = consume("PROX")
            op_raw = tok[1]
            if "/" in op_raw:
                op, num = op_raw.split("/", 1)
                try:
                    n = int(num)
                except Exception:
                    n = 0
            else:
                op = op_raw
                n = 0
            child = parse_atom()
            return _Prox(op, n, child=child)
        if t[0] == "LPAREN":
            consume("LPAREN")
            node = parse_or()
            if peek() and peek()[0] == "RPAREN":
                consume("RPAREN")
                return node
            raise ValueError("Missing closing parenthesis")
        if t[0] == "FIELD_PHRASE":
            consume("FIELD_PHRASE")
            fields, phrase = t[1]
            # If the inner phrase contains logical operators or nested
            # expressions, compile it as a sub-expression and apply the
            # field list to all contained atoms.
            if any(
                tok in phrase
                for tok in (" AND ", " OR ", " NOT ", "(", ")", "W/", "PRE/")
            ):
                # Normalize common quoting issues before attempting to parse
                inner = phrase
                # Replace doubled single-quotes often used in exported lists (e.g. Fisher''s)
                inner = inner.replace("''", "'")
                # Normalize smart quotes to simple quotes
                inner = (
                    inner.replace("“", '"')
                    .replace("”", '"')
                    .replace("‘", "'")
                    .replace("’", "'")
                )
                # Collapse multiple spaces
                inner = re.sub(r"\s+", " ", inner)
                # try to compile sub-expression to AST; if that fails,
                # fall back to OR-of-literals using quoted phrases inside.
                try:
                    sub_root = compile_expression(inner, return_ast=True)
                except Exception:
                    # extract quoted literals including escaped content
                    lits = []
                    for a in re.findall(r'"((?:[^"\\]|\\.)*)"', inner):
                        lits.append(a.replace('\\"', '"'))
                    for b in re.findall(r"'((?:[^'\\]|\\.)*)'", inner):
                        lits.append(b.replace("\\'", "'"))
                    # If there are many comma/semicolon-separated items without quotes,
                    # try to split on common separators as a last-ditch extraction.
                    if not lits:
                        parts = re.split(r"[,;]\\s*", inner)
                        # keep only reasonably short fragments
                        for p in parts:
                            p = p.strip()
                            if 2 <= len(p) <= 80:
                                lits.append(p)
                    if lits:
                        nodes = [_Atom(p, fields) for p in lits]
                        return _OrList(nodes)
                    # final fallback: treat full inner as single atom
                    return _Atom(inner, fields)

                def _apply_fields(node):
                    # mutate _Atom children to set fields
                    if isinstance(node, _Atom):
                        node.fields = fields
                    elif isinstance(node, (_And, _Or, _AndList, _OrList)):
                        # binary nodes
                        try:
                            node.left
                            _apply_fields(node.left)
                            node.right
                            _apply_fields(node.right)
                        except Exception:
                            for c in getattr(node, "children", []):
                                _apply_fields(c)
                    elif isinstance(node, _Not):
                        _apply_fields(node.child)
                    elif isinstance(node, _Prox):
                        if node.left is not None:
                            _apply_fields(node.left)
                        if node.right is not None:
                            _apply_fields(node.right)
                        if node.child is not None:
                            _apply_fields(node.child)

                _apply_fields(sub_root)
                return sub_root
            return _Atom(phrase, fields)
        if t[0] == "PHRASE":
            consume("PHRASE")
            phrase = t[1]
            if phrase[0] in '"\'"':
                phrase = phrase[1:-1]
            return _Atom(phrase, None)
        # Also allow OP tokens as invalid here
        raise ValueError(f"Unexpected token in atom: {t}")

    def parse_and():
        # collect a list of AND-terms (and PROX-treated pairs)
        terms = [parse_atom()]
        while True:
            p = peek()
            if p and p[0] == "OP" and p[1] == "AND":
                consume("OP")
                terms.append(parse_atom())
                continue
            if p and p[0] == "PROX":
                tok = consume("PROX")
                op_raw = tok[1]
                if "/" in op_raw:
                    op, num = op_raw.split("/", 1)
                    try:
                        n = int(num)
                    except Exception:
                        n = 0
                else:
                    op = op_raw
                    n = 0
                right = parse_atom()
                # attach the proximity as a combined node replacing the last term
                left = terms.pop()
                terms.append(_Prox(op, n, left=left, right=right))
                continue
            break
        if len(terms) == 1:
            return terms[0]
        return _AndList(terms)

    def parse_or():
        terms = [parse_and()]
        while peek() and peek()[0] == "OP" and peek()[1] == "OR":
            consume("OP")
            terms.append(parse_and())
        if len(terms) == 1:
            return terms[0]
        return _OrList(terms)

    root = parse_or()

    if idx != len(tokens):
        raise ValueError("Unexpected trailing tokens in expression")

    # Iterative cycle detection / visit to avoid deep recursion
    def _iterative_check(root_node):
        seen = set()
        stack = [root_node]
        while stack:
            node = stack.pop()
            nid = id(node)
            if nid in seen:
                # already visited this node; skip
                continue
            seen.add(nid)
            # push children
            if isinstance(node, (_And, _Or)):
                stack.append(node.right)
                stack.append(node.left)
            elif isinstance(node, _AndList) or isinstance(node, _OrList):
                for c in node.children:
                    stack.append(c)
            elif isinstance(node, _Not):
                stack.append(node.child)
            elif isinstance(node, _Prox):
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)
                if node.child is not None:
                    stack.append(node.child)
            # _Atom: no children

    _iterative_check(root)

    if return_ast:
        return root
    return _make_predicate_from_ast(root)


def tokenize_expression(expr: str):
    """Return list of tokens produced by the internal tokenizer (for debugging)."""
    # Reuse the robust scanning tokenizer used in compile_expression
    tokens = []
    pos = 0
    length = len(expr)

    def _match_op_at(p):
        m = re.match(r"\s*(AND|OR|NOT)\b", expr[p:], flags=re.IGNORECASE)
        if m:
            return ("OP", m.group(1).upper(), p + m.end())
        return None

    def _match_prox_at(p):
        m = re.match(r"\s*([A-Z]+/[0-9]+)", expr[p:], flags=re.IGNORECASE)
        if m:
            return ("PROX", m.group(1).upper(), p + m.end())
        return None

    def _match_paren_at(p):
        m = re.match(r"\s*([()])", expr[p:])
        if m:
            return (
                ("LPAREN" if m.group(1) == "(" else "RPAREN"),
                m.group(1),
                p + m.end(),
            )
        return None

    def _match_quoted_at(p):
        m = re.match(r"\s*(\"(?:[^\\\"]|\\.)*\"|'(?:[^\\']|\\.)*')", expr[p:])
        if m:
            return ("PHRASE", m.group(1), p + m.end())
        return None

    def _find_matching_paren(start_idx):
        i = start_idx
        depth = 0
        in_quote = None
        while i < length:
            ch = expr[i]
            if in_quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_quote:
                    in_quote = None
                i += 1
                continue
            if ch in ('"', "'"):
                in_quote = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1

    while pos < length:
        if expr[pos].isspace():
            pos += 1
            continue
        m = _match_op_at(pos)
        if m:
            tokens.append((m[0], m[1]))
            pos = m[2]
            continue
        m = _match_prox_at(pos)
        if m:
            tokens.append((m[0], m[1]))
            pos = m[2]
            continue
        m = _match_paren_at(pos)
        if m:
            tokens.append((m[0], m[1]))
            pos = m[2]
            continue
        m = _FIELD_LIST_PATTERN.match(expr[pos:])
        if m:
            fields = _normalize_fields(m.group(1))
            paren_pos = pos + m.end() - 1
            end_pos = _find_matching_paren(paren_pos)
            if end_pos == -1:
                raise ValueError("Unmatched parenthesis in FIELD(...) clause")
            inner = expr[paren_pos + 1 : end_pos].strip()
            phrase = _unwrap_single_quoted_literal(inner)
            tokens.append(("FIELD_PHRASE", (fields, phrase)))
            pos = end_pos + 1
            continue
        m = _match_quoted_at(pos)
        if m:
            tokens.append((m[0], m[1]))
            pos = m[2]
            continue
        m = re.match(r"\s*([^\s()]+)", expr[pos:])
        if m:
            tokens.append(("PHRASE", m.group(1)))
            pos += m.end()
            continue
        raise ValueError(f"Unable to tokenize at: {expr[pos:pos+20]!r}")
    return tokens


# Load and compile all SDG mapping expressions (SDG01..SDG17) from the
# Provide an object-oriented classifier so mappings are loaded once and
# multiple classifications can be performed efficiently.
sdg_dir = os.path.join(os.path.dirname(__file__), "SDG Queries")


class SDGClassifier:
    """Classifier for Elsevier-style SDG mapping expressions.

    Parameters
    ----------
    sdg_dir_path:
        Directory containing the SDG01.txt .. SDG17.txt mapping files.
        Defaults to the bundled `SDG Queries/` folder next to this module.
    embedder:
        Optional callable that accepts a list of strings and returns a list of
        vector-like embeddings. Provide this when using `similarity_method='embed'`
        or `similarity_method='embed_ngram'`.
    embedding_model_name:
        Optional sentence-transformers model id. If provided and `embedder` is
        not set, the classifier will try to load that model lazily.
    use_faiss:
        Enable optional FAISS-backed nearest-neighbor search for embedding-based
        matching. Requires `faiss` / `faiss-cpu` to be installed.
    cache_dir:
        Optional path used when persisting the computed phrase index with
        `save_phrase_index(...)` / `load_phrase_index(...)`. If this is a `.pkl`
        file path, it is treated as the phrase index file itself; otherwise it is
        treated as a cache directory.
    auto_load_cache:
        If `True`, attempt to load a previously saved phrase index and FAISS
        indices from `cache_dir` / `cache_faiss_dir` during initialization.
    auto_save_cache:
        If `True`, persist the computed phrase index and FAISS indices after
        they are built.
    cache_faiss_dir:
        Optional directory for persisted FAISS index files. Used together with
        `cache_dir` when saving or loading FAISS indices.
    """

    def __init__(
        self,
        sdg_dir_path: Optional[str] = None,
        *,
        embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
        embedding_model_name: Optional[str] = None,
        use_faiss: bool = False,
        cache_dir: Optional[str] = None,
        auto_load_cache: bool = False,
        auto_save_cache: bool = False,
        cache_faiss_dir: Optional[str] = None,
    ):
        self.sdg_dir = sdg_dir_path or sdg_dir
        if not os.path.isdir(self.sdg_dir):
            raise RuntimeError(f"SDG queries directory not found: {self.sdg_dir}")
        self.predicates: List[Callable[[str, str, Union[str, Iterable[str]]], bool]] = []
        self.ast_roots: Dict[int, Optional[_Node]] = {}
        self.phrases: Dict[int, List[str]] = {}
        self.fallback_sdgs: List[tuple] = []
        self.embedder = embedder
        self.embedding_model_name = embedding_model_name
        self.use_faiss = use_faiss
        self.auto_save_cache = auto_save_cache
        # cache directory to save/load persisted artifacts (phrase_index, faiss)
        self.cache_dir = cache_dir
        self.cache_faiss_dir = cache_faiss_dir
        self.phrase_embeddings: Dict[int, List] = {}
        # optimized index: per-SDG list of (length, phrase, embedding)
        self.phrase_index: Dict[int, List] = {}
        # optional FAISS indices: {sdg: {length: faiss_index}}
        self.faiss_index: Dict[int, Dict[int, object]] = {}
        self._load_mappings()

        # optionally auto-load persisted phrase_index and FAISS indices
        if auto_load_cache:
            # determine phrase index file path
            if cache_dir and isinstance(cache_dir, str) and cache_dir.endswith('.pkl'):
                phrase_file = cache_dir
                faiss_dir = cache_faiss_dir or (os.path.dirname(phrase_file) if os.path.dirname(phrase_file) else None)
            else:
                base_cache = cache_dir or os.path.join(self.sdg_dir, "cache")
                phrase_file = os.path.join(base_cache, "phrase_index.pkl")
                faiss_dir = cache_faiss_dir or os.path.join(base_cache, "faiss")
            try:
                if phrase_file and os.path.exists(phrase_file):
                    self.load_phrase_index(phrase_file)
            except Exception:
                # ignore cache load failures and proceed with fresh computation
                pass
            try:
                if faiss_dir and os.path.isdir(faiss_dir):
                    self.load_faiss_indices(faiss_dir)
            except Exception:
                pass

    def _load_mappings(self):
        missing = []
        for i in range(1, 18):
            candidates = [f"SDG{str(i).zfill(2)}.txt", f"SDG{i}.txt"]
            found = None
            for name in candidates:
                path = os.path.join(self.sdg_dir, name)
                if os.path.exists(path):
                    found = path
                    break
            if not found:
                missing.append(i)
                continue
            with open(found, "r", encoding="utf-8") as fh:
                expr = fh.read()
            phrases_set = set()
            for a in re.findall(r'"([^"\\]+)"', expr):
                phrases_set.add(a)
            for b in re.findall(r"'([^'\\]+)'", expr):
                phrases_set.add(b)
            # Normalize certain multi-word quoted phrases into explicit
            # proximity expressions to avoid overly loose regex matches.
            # Example: "sustainable* manag*" -> "sustainable* W/3 manag*"
            # Filter out quoted operator-like tokens (e.g. " OR ") which
            # originate from exported query text and are not real phrases.
            op_pat = re.compile(r"^\s*(AND|OR|NOT|\(|\))\s*$", flags=re.IGNORECASE)
            for p in list(phrases_set):
                if op_pat.match(p):
                    phrases_set.discard(p)
            # Normalize whitespace in quoted phrases and discard
            # operator-like quoted tokens. Keep multi-word wildcard
            # phrases as-is — wildcard matching is restricted above
            # so wildcards won't cross word boundaries.
            for p in list(phrases_set):
                if not p:
                    phrases_set.discard(p)
                    continue
                if op_pat.match(p):
                    phrases_set.discard(p)
                    continue
                # collapse internal whitespace in stored phrases
                if re.search(r"\s+", p):
                    norm = re.sub(r"\s+", " ", p).strip()
                    if norm != p:
                        phrases_set.discard(p)
                        phrases_set.add(norm)
            self.phrases[i] = sorted(phrases_set)
            try:
                root = compile_expression(expr, return_ast=True)
                self.ast_roots[i] = root
                self.predicates.append(_make_predicate_from_ast(root))
            except Exception:
                def make_phrase_pred(phrases):
                    if not phrases:
                        return lambda t, a, k: False
                    escaped = [re.escape(p) for p in phrases]
                    pat = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)

                    def pred(t, a, k):
                        kws = ""
                        if k is None:
                            kws = ""
                        elif isinstance(k, str):
                            kws = k
                        else:
                            kws = " ".join(k)
                        text = " ".join(filter(None, [t or "", a or "", kws or ""]))
                        return bool(pat.search(text))

                    return pred

                self.predicates.append(make_phrase_pred(phrases_set))
                self.ast_roots[i] = None
                self.fallback_sdgs.append((i, found))
                continue
        if missing:
            raise RuntimeError(f"Missing SDG mapping files for: {missing}")

        # if an embedder was provided or a model name given, compute phrase embeddings
        if self.embedder or self.embedding_model_name:
            # lazy-load sentence-transformers if a model name was provided
            if self.embedding_model_name and not self.embedder:
                try:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(self.embedding_model_name)

                    def _model_embed(texts: List[str]):
                        arr = model.encode(texts, convert_to_numpy=False)
                        return [list(a) for a in arr]

                    self.embedder = _model_embed
                except Exception:
                    # failed to load model; disable embedding support
                    self.embedder = None
            if self.embedder:
                for sdg, phs in self.phrases.items():
                    if phs:
                        try:
                            embs = self.embedder(phs)
                            self.phrase_embeddings[sdg] = embs
                            # build optimized index for quick length-based access
                            tuples = []
                            for p, e in zip(phs, embs):
                                tuples.append((len(p.split()), p, e))
                            self.phrase_index[sdg] = tuples
                        except Exception:
                            self.phrase_embeddings[sdg] = []
                            self.phrase_index[sdg] = []
                # optionally build FAISS indices grouped by phrase length
                if self.use_faiss:
                    try:
                        import faiss
                        import numpy as _np

                        for sdg, tuples in self.phrase_index.items():
                            length_groups = {}
                            for L, p, e in tuples:
                                length_groups.setdefault(L, []).append(e)
                            idxs = {}
                            for L, vecs in length_groups.items():
                                try:
                                    arr = _np.asarray(vecs, dtype=_np.float32)
                                    # normalize for inner-product == cosine similarity
                                    norms = _np.linalg.norm(arr, axis=1, keepdims=True)
                                    norms[norms == 0] = 1.0
                                    arr = arr / norms
                                    d = arr.shape[1]
                                    index = faiss.IndexFlatIP(d)
                                    index.add(arr)
                                    idxs[L] = index
                                except Exception:
                                    idxs[L] = None
                            self.faiss_index[sdg] = idxs
                    except Exception:
                        # If faiss not available, silently skip building indices
                        self.faiss_index = {}

                if self.auto_save_cache:
                    self.persist_cache()

        # Persistence is explicit: call `save_phrase_index` / `save_faiss_indices`
        # to persist computed embeddings/indices. Automatic writing at init
        # has been intentionally removed to avoid side-effects on import.

    def _resolve_cache_paths(self):
        if self.cache_dir and isinstance(self.cache_dir, str) and self.cache_dir.endswith('.pkl'):
            phrase_file = self.cache_dir
            faiss_dir = self.cache_faiss_dir or (os.path.dirname(phrase_file) if os.path.dirname(phrase_file) else None)
        else:
            base_cache = self.cache_dir or os.path.join(self.sdg_dir, "cache")
            phrase_file = os.path.join(base_cache, "phrase_index.pkl")
            faiss_dir = self.cache_faiss_dir or os.path.join(base_cache, "faiss")
        return phrase_file, faiss_dir

    def persist_cache(self):
        """Persist the current phrase index and FAISS indices using the configured cache paths."""
        phrase_file, faiss_dir = self._resolve_cache_paths()
        saved_any = False
        if phrase_file:
            saved_any = self.save_phrase_index(phrase_file) or saved_any
        if faiss_dir and self.use_faiss:
            saved_any = self.save_faiss_indices(faiss_dir) or saved_any
        return saved_any

    def save_phrase_index(self, path: str):
        """Save the computed phrase_index (including embeddings) to a pickle file."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, "wb") as fh:
                pickle.dump({"phrase_index": self.phrase_index, "phrases": self.phrases}, fh, protocol=4)
            return True
        except Exception:
            return False

    def load_phrase_index(self, path: str):
        """Load a previously saved phrase_index. Does not validate dimensions."""
        p = pathlib.Path(path)
        if not p.exists():
            return False
        try:
            with open(p, "rb") as fh:
                data = pickle.load(fh)
            self.phrase_index = data.get("phrase_index", {})
            self.phrases = data.get("phrases", {})
            # rebuild phrase_embeddings mapping from phrase_index
            ph_embs = {}
            for sdg, tuples in self.phrase_index.items():
                ph_embs[sdg] = [t[2] for t in tuples]
            self.phrase_embeddings = ph_embs
            return True
        except Exception:
            return False

    def save_faiss_indices(self, dirpath: str):
        """Persist any built FAISS indices to files under `dirpath`.

        Files are written as: `{dirpath}/faiss_sdg{sdg}_len{L}.index`.
        """
        try:
            import faiss
        except Exception:
            return False
        p = pathlib.Path(dirpath)
        p.mkdir(parents=True, exist_ok=True)
        try:
            wrote_any = False
            for sdg, idx_map in (self.faiss_index or {}).items():
                for L, index in idx_map.items():
                    if index is None:
                        continue
                    fname = p / f"faiss_sdg{sdg}_len{L}.index"
                    try:
                        faiss.write_index(index, str(fname))
                        wrote_any = True
                    except Exception:
                        # ignore single failures
                        continue
            return wrote_any
        except Exception:
            return False

    def load_faiss_indices(self, dirpath: str):
        """Load FAISS indices from `dirpath` if present (reverse of save)."""
        try:
            import faiss
        except Exception:
            return False
        p = pathlib.Path(dirpath)
        if not p.exists():
            return False
        loaded = {}
        for f in p.iterdir():
            name = f.name
            m = re.match(r"faiss_sdg(\d+)_len(\d+)\.index$", name)
            if not m:
                continue
            sdg = int(m.group(1))
            L = int(m.group(2))
            try:
                idx = faiss.read_index(str(f))
            except Exception:
                idx = None
            loaded.setdefault(sdg, {})[L] = idx
        if loaded:
            self.faiss_index = loaded
            return True
        return False

    def classify(
        self,
        title: Optional[str],
        abstract: Optional[str],
        keywords: Optional[Union[str, Iterable[str]]],
        similarity_threshold: Optional[Union[float, Dict[int, float]]] = None,
        similarity_method: Optional[str] = None,
    ) -> List[int]:
        """Classify one article against all 17 SDG mappings.

        Parameters
        ----------
        title:
            Article title text. May be `None`.
        abstract:
            Article abstract text. May be `None`.
        keywords:
            Keyword text. Accepts either a single string or an iterable of
            keyword strings. May be `None`.
        similarity_threshold:
            Either a float in the range 0.0 to 1.0 used for approximate matching
            (applied to all SDGs), or a mapping `{sdg_index: float}` to provide
            a different threshold per SDG. When provided, exact predicate
            matches are supplemented with a SequenceMatcher-based word
            similarity fallback using the resolved per-SDG threshold.
        similarity_method:
            Controls embedding-based matching after exact matching.
            Supported values are:
            - `None` (default): exact matching plus the optional
              `similarity_threshold` fallback.
            - `"embed"`: whole-document bi-encoder matching.
            - `"embed_ngram"`: sliding n-gram bi-encoder matching.

        Returns
        -------
        List[int]
            A 17-element list of 0/1 flags, where position 0 corresponds to
            SDG1 and position 16 corresponds to SDG17.
        """
        results: List[int] = []
        cosine = _cosine_similarity_fn()

        # prepare sliding windows for embed_ngram if requested
        doc_windows_by_len: Dict[int, List[str]] = {}
        window_embs_cache: Dict[int, List] = {}
        if (
            similarity_method == "embed_ngram"
            and similarity_threshold is not None
            and self.embedder
            and isinstance(self.phrase_embeddings, dict)
        ):
            try:
                doc_text = _build_doc_text(title, abstract, keywords)
            except Exception:
                doc_text = ""
            if doc_text:
                doc_tokens = doc_text.split()
                unique_lengths = set()
                for phs in self.phrases.values():
                    for p in phs:
                        unique_lengths.add(len(p.split()))
                for L in unique_lengths:
                    if L <= 0:
                        continue
                    if len(doc_tokens) < L:
                        continue
                    windows = [" ".join(doc_tokens[i : i + L]) for i in range(len(doc_tokens) - L + 1)]
                    if windows:
                        doc_windows_by_len[L] = windows

        for pred in self.predicates:
            sdg_index = len(results) + 1
            # resolve per-SDG similarity threshold: may be a float or a dict
            if isinstance(similarity_threshold, dict):
                sdg_sim = similarity_threshold.get(sdg_index)
            else:
                sdg_sim = similarity_threshold

            try:
                hit = _call_predicate(
                    pred,
                    title,
                    abstract,
                    keywords,
                    similarity_threshold=sdg_sim,
                )
            except Exception:
                hit = False

            # embedding-based fallback: whole-document bi-encoder
            if (
                not hit
                and similarity_method == "embed"
                and sdg_sim is not None
                and self.embedder
                and isinstance(self.phrase_embeddings, dict)
            ):
                try:
                    doc_text = _build_doc_text(title, abstract, keywords)
                except Exception:
                    doc_text = ""
                if doc_text:
                    try:
                        vecs = self.embedder([doc_text])
                        if vecs:
                            doc_vec = vecs[0]
                            ph_embs = self.phrase_embeddings.get(sdg_index, [])
                            for pvec in ph_embs:
                                if cosine(doc_vec, pvec) >= sdg_sim:
                                    hit = True
                                    break
                    except Exception:
                        pass

            # sliding n-gram bi-encoder matching (compare phrase embeddings
            # against window embeddings of matching token-length)
            if (
                not hit
                and similarity_method == "embed_ngram"
                and sdg_sim is not None
                and self.embedder
                and isinstance(self.phrase_embeddings, dict)
            ):
                phs = self.phrases.get(sdg_index, [])
                ph_embs = self.phrase_embeddings.get(sdg_index, [])
                if phs and ph_embs:
                    # use optimized phrase_index if available
                    tuples = self.phrase_index.get(sdg_index)
                    if tuples is None:
                        tuples = list(zip((len(p.split()) for p in phs), phs, ph_embs))
                    for L, phrase, pvec in tuples:
                        if L <= 0:
                            continue
                        windows = doc_windows_by_len.get(L)
                        if not windows:
                            continue
                        # Try FAISS search if available for this SDG/length
                        faiss_used = False
                        try:
                            sdg_idxs = getattr(self, "faiss_index", {}) or {}
                            sdg_map = sdg_idxs.get(sdg_index, {})
                            index = sdg_map.get(L) if sdg_map else None
                        except Exception:
                            index = None
                        if index is not None:
                            try:
                                import numpy as _np

                                wembs = window_embs_cache.get(L)
                                if wembs is None:
                                    try:
                                        wembs = self.embedder(windows)
                                    except Exception:
                                        wembs = []
                                    window_embs_cache[L] = wembs
                                if wembs:
                                    arr = _np.asarray(wembs, dtype=_np.float32)
                                    norms = _np.linalg.norm(arr, axis=1, keepdims=True)
                                    norms[norms == 0] = 1.0
                                    arr = arr / norms
                                    D, I = index.search(arr, 1)
                                    if (D >= float(sdg_sim)).any():
                                        hit = True
                                        faiss_used = True
                            except Exception:
                                faiss_used = False
                        if faiss_used:
                            break

                        # fallback to brute-force comparison using embedder
                        if L not in window_embs_cache:
                            try:
                                window_embs_cache[L] = self.embedder(windows)
                            except Exception:
                                window_embs_cache[L] = []
                        wembs = window_embs_cache.get(L, [])
                        for wvec in wembs:
                            if cosine(pvec, wvec) >= sdg_sim:
                                hit = True
                                break
                        if hit:
                            break

            results.append(1 if hit else 0)

        return results

    def matched_phrases(
        self,
        title: Optional[str],
        abstract: Optional[str],
        keywords: Optional[Union[str, Iterable[str]]],
        similarity_threshold: Optional[Union[float, Dict[int, float]]] = None,
    ) -> Dict[int, List[str]]:
        """Return matched phrases per SDG for this classifier instance.

        `similarity_threshold` may be a float or a dict mapping SDG index -> float
        to control per-SDG fuzzy matching behavior.
        """

        text = _combined_article_text(title, abstract, keywords)
        out: Dict[int, List[str]] = {}
        if not text:
            for i in range(1, 18):
                out[i] = []
            return out

        for sdg_index in range(1, 18):
            if isinstance(similarity_threshold, dict):
                sdg_sim = similarity_threshold.get(sdg_index)
            else:
                sdg_sim = similarity_threshold
            hits: List[str] = []
            for phrase in self.phrases.get(sdg_index, []):
                if _phrase_matches_text(phrase, text, sdg_sim):
                    hits.append(phrase)
            out[sdg_index] = hits
        return out

    def debug_sdg_branches(
        self,
        sdg_index: int,
        title: Optional[str],
        abstract: Optional[str],
        keywords: Optional[Union[str, Iterable[str]]],
        similarity_threshold: Optional[Union[float, Dict[int, float]]] = None,
        matched_only: bool = False,
    ) -> Dict[str, Any]:
        """Inspect top-level branch matches and true atoms for one SDG.

        Returns a dict with:
        - sdg: SDG index (1..17)
        - matched: whether any top-level branch matched
        - matched_branch_indices: list of matched top-level branch indices
        - branches: list of branch details with true atom phrases

        When `matched_only=True`, only matched top-level branches are returned
        under `branches`.
        """
        if not (1 <= sdg_index <= 17):
            raise ValueError("sdg_index must be in 1..17")

        root = self.ast_roots.get(sdg_index)
        if root is None:
            return {
                "sdg": sdg_index,
                "matched": False,
                "matched_branch_indices": [],
                "branches": [],
                "note": "No AST available for this SDG (fallback parser was used).",
            }

        if keywords is None:
            kws = ""
        elif isinstance(keywords, str):
            kws = keywords
        else:
            kws = " ".join(str(k) for k in keywords if k)

        if isinstance(similarity_threshold, dict):
            sdg_sim = similarity_threshold.get(sdg_index)
        else:
            sdg_sim = similarity_threshold

        def _collect_atoms(node: _Node, out: List[_Atom]) -> None:
            if isinstance(node, _Atom):
                out.append(node)
                return
            if isinstance(node, (_And, _Or)):
                _collect_atoms(node.left, out)
                _collect_atoms(node.right, out)
                return
            if isinstance(node, (_AndList, _OrList)):
                for c in node.children:
                    _collect_atoms(c, out)
                return
            if isinstance(node, _Not):
                _collect_atoms(node.child, out)
                return
            if isinstance(node, _Prox):
                if node.left is not None:
                    _collect_atoms(node.left, out)
                if node.right is not None:
                    _collect_atoms(node.right, out)
                if node.child is not None:
                    _collect_atoms(node.child, out)

        top_children = root.children if isinstance(root, _OrList) else [root]
        branches: List[Dict[str, Any]] = []
        matched_idx: List[int] = []

        for idx, branch in enumerate(top_children):
            try:
                branch_hit = bool(
                    branch.eval(title or "", abstract or "", kws, similarity_threshold=sdg_sim)
                )
            except Exception:
                branch_hit = False

            atoms: List[_Atom] = []
            _collect_atoms(branch, atoms)
            true_atoms: List[str] = []
            seen: set = set()
            for a in atoms:
                try:
                    if a.eval(title or "", abstract or "", kws, similarity_threshold=sdg_sim):
                        if a.phrase not in seen:
                            true_atoms.append(a.phrase)
                            seen.add(a.phrase)
                except Exception:
                    continue

            if branch_hit:
                matched_idx.append(idx)
            if (not matched_only) or branch_hit:
                branches.append(
                    {
                        "index": idx,
                        "matched": branch_hit,
                        "true_atoms": true_atoms,
                        "true_atom_count": len(true_atoms),
                        "atom_count": len(atoms),
                    }
                )

        return {
            "sdg": sdg_index,
            "matched": bool(matched_idx),
            "matched_branch_indices": matched_idx,
            "branches": branches,
        }

# instantiate a default classifier for backward compatibility (loaded once)
DEFAULT_CLASSIFIER = SDGClassifier(sdg_dir)
# module-level aliases for backward compatibility
SDG_PREDICATES = DEFAULT_CLASSIFIER.predicates
SDG_PHRASES = DEFAULT_CLASSIFIER.phrases
fallback_sdgs = DEFAULT_CLASSIFIER.fallback_sdgs


def classify_sdgs(
    title: Optional[str],
    abstract: Optional[str],
    keywords: Optional[Union[str, Iterable[str]]],
    similarity_threshold: Optional[Union[float, Dict[int, float]]] = None,
    similarity_method: Optional[str] = None,
) -> List[int]:
    """Return a 17-element list of 0/1 flags indicating SDG membership.

    Position 0 -> SDG1, position 16 -> SDG17.

    If `similarity_threshold` is provided, atoms that do not match exactly are
    also compared using a word-level similarity fallback. Values should be in
    the range 0.0 to 1.0.

        `similarity_method` controls the embedding-based fallback mode:
        - `None` (default): exact expression matching plus optional word-level
            similarity fallback.
        - `"embed"`: whole-document bi-encoder matching against phrase embeddings.
        - `"embed_ngram"`: sliding n-gram bi-encoder matching against token-length
            windows in the document.
    """
    # Forward to the default classifier instance which supports embedding
    # based methods such as 'embed' and 'embed_ngram'. This keeps
    # backward-compatibility while exposing the extended API.
    try:
        return DEFAULT_CLASSIFIER.classify(
            title, abstract, keywords, similarity_threshold=similarity_threshold, similarity_method=similarity_method
        )
    except Exception:
        # fall back to predicate-only behavior if something goes wrong
        results: List[int] = []
        for i, pred in enumerate(SDG_PREDICATES, start=1):
            # resolve per-SDG similarity threshold if a mapping provided
            if isinstance(similarity_threshold, dict):
                sdg_sim = similarity_threshold.get(i)
            else:
                sdg_sim = similarity_threshold
            try:
                hit = _call_predicate(
                    pred,
                    title,
                    abstract,
                    keywords,
                    similarity_threshold=sdg_sim,
                )
            except Exception:
                hit = False
            results.append(1 if hit else 0)
        return results


def matched_phrases_per_sdg(
    title: Optional[str],
    abstract: Optional[str],
    keywords: Optional[Union[str, Iterable[str]]],
    similarity_threshold: Optional[Union[float, Dict[int, float]]] = None,
) -> Dict[int, List[str]]:
    """Return matched phrases per SDG using the default classifier instance.

    `similarity_threshold` may be a float or a dict mapping SDG index -> float
    to control per-SDG fuzzy matching behavior.
    """

    return DEFAULT_CLASSIFIER.matched_phrases(
        title, abstract, keywords, similarity_threshold=similarity_threshold
    )


def debug_sdg_branches(
    sdg_index: int,
    title: Optional[str],
    abstract: Optional[str],
    keywords: Optional[Union[str, Iterable[str]]],
    similarity_threshold: Optional[Union[float, Dict[int, float]]] = None,
    matched_only: bool = False,
) -> Dict[str, Any]:
    """Debug top-level branch matches and true atoms for one SDG.

    Set `matched_only=True` to return only matched top-level branches.
    """
    return DEFAULT_CLASSIFIER.debug_sdg_branches(
        sdg_index,
        title,
        abstract,
        keywords,
        similarity_threshold=similarity_threshold,
        matched_only=matched_only,
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "SDG classifier utility: classify text or inspect compiled mappings."
        )
    )
    parser.add_argument("-t", "--title", help="Article title text")
    parser.add_argument("-a", "--abstract", help="Article abstract text")
    parser.add_argument(
        "-k",
        "--keywords",
        action="append",
        help="Keyword(s); may be provided multiple times",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run a small built-in demo classification"
    )
    parser.add_argument(
        "--list-sdgs", action="store_true", help="List SDGs and number of extracted phrases"
    )
    parser.add_argument(
        "--check-fallbacks",
        action="store_true",
        help="Show SDGs that used a simple quoted-phrase fallback at import",
    )
    parser.add_argument(
        "--debug-sdg",
        type=int,
        help="Debug one SDG: print top-level matched branches and true atoms",
    )
    parser.add_argument(
        "--debug-matched-only",
        action="store_true",
        help="With --debug-sdg, print only matched top-level branches",
    )
    args = parser.parse_args()

    if args.demo:
        sample_title = "Reducing poverty and hunger in rural communities"
        sample_abstract = (
            "We investigate social protection and safety nets to tackle food insecurity "
            "and extreme poverty in low income regions."
        )
        sample_keywords = ["poverty", "social protection"]
        # demo: default classification (no per-SDG similarity thresholds)
        vec = classify_sdgs(sample_title, sample_abstract, sample_keywords)
        print("SDG vector (default):", vec)
        matched = [i + 1 for i, v in enumerate(vec) if v]
        print("Matched SDGs (default):", matched or "(none)")

        # demo: per-SDG similarity threshold map example
        # provide a dict mapping SDG index -> similarity threshold (0.0-1.0)
        per_sdg_thresh = {1: 1, 14: 0.92, 17: 0.90}
        vec2 = classify_sdgs(sample_title, sample_abstract, sample_keywords, similarity_threshold=per_sdg_thresh, similarity_method="embed")
        print("SDG vector (per-SDG thresholds):", vec2)
        matched2 = [i + 1 for i, v in enumerate(vec2) if v]
        print("Matched SDGs (per-SDG thresholds):", matched2 or "(none)")
        # demo: matched phrases per SDG for debugging
        print('\nMatched phrases (default):')
        mp_def = matched_phrases_per_sdg(sample_title, sample_abstract, sample_keywords)
        print(json.dumps({k: v for k, v in mp_def.items() if v}, indent=2, ensure_ascii=False))

        print('\nMatched phrases (per-SDG thresholds):')
        mp_per = matched_phrases_per_sdg(sample_title, sample_abstract, sample_keywords, similarity_threshold=per_sdg_thresh)
        print(json.dumps({k: v for k, v in mp_per.items() if v}, indent=2, ensure_ascii=False))
        sys.exit(0)

    if args.check_fallbacks:
        if fallback_sdgs:
            print("Fallback SDGs used during import:")
            for i, p in fallback_sdgs:
                print(f" - SDG{i}: {p}")
        else:
            print("No fallback SDGs detected (all mappings compiled).")
        sys.exit(0)

    if args.list_sdgs:
        for i in range(1, 18):
            ph = SDG_PHRASES.get(i, [])
            print(f"SDG{i}: {len(ph)} extracted phrases")
            if ph:
                print("  example:", ph[0])
        sys.exit(0)

    if args.debug_sdg is not None:
        report = debug_sdg_branches(
            args.debug_sdg,
            args.title,
            args.abstract,
            args.keywords,
            matched_only=args.debug_matched_only,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(0)

    if args.title or args.abstract or args.keywords:
        kws = None
        if args.keywords:
            kws = args.keywords if len(args.keywords) > 1 else args.keywords[0]
        vec = classify_sdgs(args.title, args.abstract, kws)
        print("SDG vector:", vec)
        matched = [str(i + 1) for i, v in enumerate(vec) if v]
        print("Matched SDGs:", ", ".join(matched) if matched else "None")
        sys.exit(0)

    parser.print_help()
    sys.exit(0)