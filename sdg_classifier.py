import re
import os
from typing import Iterable, Union, List, Dict, Callable, Optional

# --- Expression-based mapping support ---------------------------------
# Basic grammar supports atoms of the form FIELD1-FIELD2("phrase") or
# "phrase" (quoted) and boolean operators AND, OR and parentheses. Fields
# can be TITLE, ABS, KW. Example: ("poverty" OR TITLE-ABS("unesco")) AND KW("social protection")

_pattern_cache: dict = {}


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
            # restore wildcard regex
            s = s.replace(placeholder, ".*")
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


class _Node:
    def eval(self, title, abstract, kws):
        raise NotImplementedError()


class _Atom(_Node):
    def __init__(self, phrase: str, fields: Union[None, list] = None):
        # fields==None means match any field (title|abstract|keywords)
        self.phrase = phrase
        self.fields = fields

    def eval(self, title, abstract, kws):
        if not self.fields:
            # any field
            pat = _get_pattern(self.phrase)
            return bool(
                pat.search(title or "")
                or pat.search(abstract or "")
                or pat.search(kws or "")
            )
        for f in self.fields:
            if _match_in_field(self.phrase, f, title, abstract, kws):
                return True
        return False


class _And(_Node):
    def __init__(self, left: _Node, right: _Node):
        self.left = left
        self.right = right

    def eval(self, title, abstract, kws):
        return self.left.eval(title, abstract, kws) and self.right.eval(
            title, abstract, kws
        )


class _Or(_Node):
    def __init__(self, left: _Node, right: _Node):
        self.left = left
        self.right = right

    def eval(self, title, abstract, kws):
        return self.left.eval(title, abstract, kws) or self.right.eval(
            title, abstract, kws
        )


class _AndList(_Node):
    def __init__(self, children):
        self.children = children

    def eval(self, title, abstract, kws):
        for c in self.children:
            if not c.eval(title, abstract, kws):
                return False
        return True


class _OrList(_Node):
    def __init__(self, children):
        self.children = children

    def eval(self, title, abstract, kws):
        for c in self.children:
            if c.eval(title, abstract, kws):
                return True
        return False


class _Not(_Node):
    def __init__(self, child: _Node):
        self.child = child

    def eval(self, title, abstract, kws):
        return not self.child.eval(title, abstract, kws)


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

    def eval(self, title, abstract, kws):
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
                # as a last resort, fall back to logical AND
                try:
                    return L.eval(title, abstract, kws) and R.eval(title, abstract, kws)
                except Exception:
                    return False
            # for non-atom children, evaluate child if present
            try:
                return self.left.eval(title, abstract, kws) and self.right.eval(
                    title, abstract, kws
                )
            except Exception:
                return False
        if self.child is not None:
            return self.child.eval(title, abstract, kws)
        return False


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
        m = re.match(r"\s*([A-Z]+(?:-[A-Z]+)*)\s*\(", expr[pos:], flags=re.IGNORECASE)
        if m:
            fields_raw = m.group(1).upper().split("-")
            norm = []
            for f in fields_raw:
                if f in ("AUTHKEY", "KEY"):
                    norm.append("KW")
                else:
                    norm.append(f)
            fields = norm
            # find the absolute position of the opening paren
            paren_pos = pos + m.end() - 1
            end_pos = _find_matching_paren(paren_pos)
            if end_pos == -1:
                raise ValueError("Unmatched parenthesis in FIELD(...) clause")
            inner = expr[paren_pos + 1 : end_pos].strip()
            # strip surrounding quotes if present
            if (inner.startswith('"') and inner.endswith('"')) or (
                inner.startswith("'") and inner.endswith("'")
            ):
                phrase = inner[1:-1]
            else:
                phrase = inner
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

    def predicate(title, abstract, keywords):
        if keywords is None:
            kws = ""
        elif isinstance(keywords, str):
            kws = keywords
        else:
            kws = " ".join(keywords)
        try:
            return root.eval(title or "", abstract or "", kws)
        except RecursionError:
            # defensively treat recursion as non-match
            return False

    if return_ast:
        return root
    return predicate


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
        m = re.match(r"\s*([A-Z]+(?:-[A-Z]+)*)\s*\(", expr[pos:], flags=re.IGNORECASE)
        if m:
            fields_raw = m.group(1).upper().split("-")
            norm = []
            for f in fields_raw:
                if f in ("AUTHKEY", "KEY"):
                    norm.append("KW")
                else:
                    norm.append(f)
            fields = norm
            paren_pos = pos + m.end() - 1
            end_pos = _find_matching_paren(paren_pos)
            if end_pos == -1:
                raise ValueError("Unmatched parenthesis in FIELD(...) clause")
            inner = expr[paren_pos + 1 : end_pos].strip()
            if (inner.startswith('"') and inner.endswith('"')) or (
                inner.startswith("'") and inner.endswith("'")
            ):
                phrase = inner[1:-1]
            else:
                phrase = inner
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
sdg_dir = os.path.join(os.path.dirname(__file__), "SDG 2023 Queries")


class SDGClassifier:
    def __init__(self, sdg_dir_path: Optional[str] = None):
        self.sdg_dir = sdg_dir_path or sdg_dir
        if not os.path.isdir(self.sdg_dir):
            raise RuntimeError(f"SDG queries directory not found: {self.sdg_dir}")
        self.predicates: List[Callable[[str, str, Union[str, Iterable[str]]], bool]] = []
        self.phrases: Dict[int, List[str]] = {}
        self.fallback_sdgs: List[tuple] = []
        self._load_mappings()

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
            self.phrases[i] = sorted(phrases_set)
            try:
                self.predicates.append(compile_expression(expr))
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
                self.fallback_sdgs.append((i, found))
                continue
        if missing:
            raise RuntimeError(f"Missing SDG mapping files for: {missing}")

    def classify(self, title: Optional[str], abstract: Optional[str], keywords: Optional[Union[str, Iterable[str]]]) -> List[int]:
        results: List[int] = []
        for pred in self.predicates:
            try:
                hit = bool(pred(title, abstract, keywords))
            except Exception:
                hit = False
            results.append(1 if hit else 0)
        return results

    def is_sdg1(self, title: Optional[str], abstract: Optional[str], keywords: Optional[Union[str, Iterable[str]]]) -> bool:
        if not self.predicates:
            raise RuntimeError("No SDG predicates available")
        return bool(self.predicates[0](title, abstract, keywords))


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
) -> List[int]:
    """Return a 17-element list of 0/1 flags indicating SDG membership.

    Position 0 -> SDG1, position 16 -> SDG17.
    """
    results: List[int] = []
    for pred in SDG_PREDICATES:
        try:
            hit = bool(pred(title, abstract, keywords))
        except Exception:
            # If a predicate errors at runtime, treat as non-match (but do
            # not silently swallow compilation-time errors which were raised
            # during import).
            hit = False
        results.append(1 if hit else 0)
    return results


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
    args = parser.parse_args()

    if args.demo:
        sample_title = "Reducing poverty and hunger in rural communities"
        sample_abstract = (
            "We investigate social protection and safety nets to tackle food insecurity "
            "and extreme poverty in low income regions."
        )
        sample_keywords = ["poverty", "social protection"]
        vec = classify_sdgs(sample_title, sample_abstract, sample_keywords)
        print("SDG vector:", vec)
        matched = [i + 1 for i, v in enumerate(vec) if v]
        print("Matched SDGs:", matched or "(none)")
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
