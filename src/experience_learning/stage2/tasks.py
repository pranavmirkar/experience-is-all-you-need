"""Function-level Python task bank with hidden tests.

Structure: *family* (a skill area) -> *template* (one kind of problem) -> *instance*
(a template with sampled parameters, so many distinct tasks per template).

Split rule: splits are assigned per **template**, never per instance. Instances of one
template are siblings (same structure, different constants) and must not straddle
train, dev and test. Each stream family has train, dev and test templates, so
retention and generalisation are both measured on a template the model never trained
on. One whole family is held out: it has no train templates and is used for transfer
evaluation.

Expected outputs come from a trusted reference solution run in-process at generation
time, then JSON-normalised. Candidates never see expected outputs (see ``sandbox``).
"""

from __future__ import annotations

import json
import math
import random
import string
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from ..schemas import Split

ENTRY = "solve"
N_TESTS = 12
N_VISIBLE = 3

STREAM_FAMILIES = ("strings", "lists", "dicts", "math", "dp")
HELDOUT_FAMILIES = ("parsing",)


@dataclass(frozen=True)
class CheckCase:
    args: tuple
    expected: Any  # JSON-normalised


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    template: str
    split: Split
    prompt: str
    reference_source: str
    tests: tuple[CheckCase, ...]
    entry_point: str = ENTRY
    n_visible: int = N_VISIBLE

    @property
    def visible(self) -> tuple[CheckCase, ...]:
        return self.tests[: self.n_visible]

    @property
    def hidden(self) -> tuple[CheckCase, ...]:
        return self.tests[self.n_visible :]


@dataclass(frozen=True)
class Spec:
    description: str
    reference: str  # source defining ``solve``
    gen: Callable[[random.Random], list]  # one argument list
    edges: tuple = ()  # argument lists always included first


_TEMPLATES: dict[str, dict[str, Callable[[random.Random], Spec]]] = {}


def template(family: str, name: str):
    def deco(fn):
        _TEMPLATES.setdefault(family, {})[name] = fn
        return fn

    return deco


# ---------------------------------------------------------------- input helpers
def _ints(r, lo, hi, nmin=0, nmax=10):
    return [r.randint(lo, hi) for _ in range(r.randint(nmin, nmax))]


def _text(r, alphabet, nmin=0, nmax=14):
    return "".join(r.choice(alphabet) for _ in range(r.randint(nmin, nmax)))


def _words(r, nmin=0, nmax=7):
    return " ".join(_text(r, string.ascii_lowercase, 1, 8) for _ in range(r.randint(nmin, nmax)))


def _strdict(r, nmax=6, lo=-5, hi=9):
    return {_text(r, "abcdef", 1, 2): r.randint(lo, hi) for _ in range(r.randint(0, nmax))}


# ---------------------------------------------------------------- strings
@template("strings", "vowel_positions")
def _(r):
    m = r.randint(2, 4)
    k = r.randint(0, m - 1)
    return Spec(
        f"Return how many characters at indices i with i % {m} == {k} are vowels (a, e, i, o, u, either case).",
        f"def solve(s):\n    return sum(1 for i, c in enumerate(s) if i % {m} == {k} and c.lower() in 'aeiou')",
        lambda r_: [_text(r_, "abcdeiouxyzAEIOU ", 0, 16)],
        edges=([""],),
    )


@template("strings", "shift_letters")
def _(r):
    k = r.randint(1, 12)
    return Spec(
        f"Shift every letter forward by {k} places in the alphabet, wrapping around and keeping case. Leave other characters unchanged.",
        "def solve(s):\n    out = []\n    for c in s:\n        if c.isalpha() and c.isascii():\n"
        f"            base = ord('a') if c.islower() else ord('A')\n            out.append(chr((ord(c) - base + {k}) % 26 + base))\n"
        "        else:\n            out.append(c)\n    return ''.join(out)",
        lambda r_: [_text(r_, "abcxyzXYZ -,.3", 0, 14)],
        edges=([""],),
    )


@template("strings", "run_length")
def _(r):
    sep = r.choice(["-", ":", "#", "|"])
    return Spec(
        f"Run-length encode the string: each maximal run of equal characters becomes the character followed by the run length, runs joined by '{sep}'. Empty string gives ''.",
        "def solve(s):\n    parts = []\n    i = 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n"
        f"        parts.append(s[i] + str(j - i))\n        i = j\n    return '{sep}'.join(parts)",
        lambda r_: [_text(r_, "aab", 0, 14)],
        edges=([""], ["aaaa"]),
    )


@template("strings", "longest_word")
def _(r):
    cap = r.randint(3, 7)
    return Spec(
        f"Split the string on spaces. Return the longest word whose length is at most {cap}; on ties return the earliest. Return '' if there is none.",
        f"def solve(s):\n    best = ''\n    for w in s.split():\n        if len(w) <= {cap} and len(w) > len(best):\n            best = w\n    return best",
        lambda r_: [_words(r_)],
        edges=([""],),
    )


@template("strings", "loose_palindrome")
def _(r):
    ign = "".join(sorted(r.sample(" ,.-'!", r.randint(1, 3))))
    return Spec(
        f"Return True if the string is a palindrome ignoring case and ignoring the characters {ign!r}.",
        f"def solve(s):\n    t = [c.lower() for c in s if c not in {ign!r}]\n    return t == t[::-1]",
        lambda r_: [r_.choice([_text(r_, "abAB ,.", 0, 8) , (lambda h: h + h[::-1])(_text(r_, "abc", 0, 5))])],
        edges=([""],),
    )


# ---------------------------------------------------------------- lists
@template("lists", "power_sum")
def _(r):
    m, p, mod = r.randint(2, 5), r.randint(2, 4), r.choice([97, 1000, 10007])
    return Spec(
        f"Return the sum of x**{p} over elements x divisible by {m}, taken modulo {mod}.",
        f"def solve(a):\n    return sum(x ** {p} for x in a if x % {m} == 0) % {mod}",
        lambda r_: [_ints(r_, -30, 30, 0, 10)],
        edges=([[]],),
    )


@template("lists", "rotate_filter")
def _(r):
    k, t = r.randint(1, 6), r.randint(-3, 5)
    return Spec(
        f"Rotate the list left by {k} positions (an empty list stays empty), then keep only elements greater than {t}.",
        f"def solve(a):\n    if not a:\n        return []\n    k = {k} % len(a)\n    b = a[k:] + a[:k]\n    return [x for x in b if x > {t}]",
        lambda r_: [_ints(r_, -8, 12, 0, 9)],
        edges=([[]],),
    )


@template("lists", "second_largest")
def _(r):
    d = r.randint(-9, 9)
    return Spec(
        f"Return the second largest distinct value, or {d} if there are fewer than two distinct values.",
        f"def solve(a):\n    u = sorted(set(a), reverse=True)\n    return u[1] if len(u) >= 2 else {d}",
        lambda r_: [_ints(r_, -6, 6, 0, 8)],
        edges=([[]], [[3, 3]]),
    )


@template("lists", "capped_prefix")
def _(r):
    c = r.randint(5, 25)
    return Spec(
        f"Return the running totals of the list, where the running total never exceeds {c}: total = min(total + x, {c}).",
        f"def solve(a):\n    out = []\n    t = 0\n    for x in a:\n        t = min(t + x, {c})\n        out.append(t)\n    return out",
        lambda r_: [_ints(r_, -5, 9, 0, 10)],
        edges=([[]],),
    )


@template("lists", "pair_count")
def _(r):
    s = r.randint(-2, 12)
    return Spec(
        f"Count pairs of positions i < j with a[i] + a[j] == {s}.",
        f"def solve(a):\n    return sum(1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] + a[j] == {s})",
        lambda r_: [_ints(r_, -3, 9, 0, 10)],
        edges=([[]],),
    )


# ---------------------------------------------------------------- dicts
@template("dicts", "group_sum")
def _(r):
    m = r.randint(2, 4)
    return Spec(
        f"The input is a list of [key, value] integer pairs. Group by key % {m}, sum the values of each group, and return a list of [group, sum] sorted by group.",
        f"def solve(pairs):\n    d = {{}}\n    for k, v in pairs:\n        d[k % {m}] = d.get(k % {m}, 0) + v\n    return [[g, d[g]] for g in sorted(d)]",
        lambda r_: [[[r_.randint(-9, 9), r_.randint(-5, 9)] for _ in range(r_.randint(0, 8))]],
        edges=([[]],),
    )


@template("dicts", "mode_smallest")
def _(r):
    d = r.choice([None, -1, 0])
    return Spec(
        f"Return the most frequent element of the list; on ties return the smallest. Return {d!r} for an empty list.",
        f"def solve(a):\n    if not a:\n        return {d!r}\n    c = {{}}\n    for x in a:\n        c[x] = c.get(x, 0) + 1\n    return min(c, key=lambda x: (-c[x], x))",
        lambda r_: [_ints(r_, -4, 6, 0, 12)],
        edges=([[]],),
    )


@template("dicts", "merge_filter")
def _(r):
    t = r.randint(-2, 6)
    return Spec(
        f"Merge two dicts by adding values for shared keys, then keep only entries whose value is at least {t}. Return the resulting dict.",
        f"def solve(a, b):\n    d = dict(a)\n    for k, v in b.items():\n        d[k] = d.get(k, 0) + v\n    return {{k: v for k, v in d.items() if v >= {t}}}",
        lambda r_: [_strdict(r_), _strdict(r_)],
        edges=([{}, {}],),
    )


@template("dicts", "invert_groups")
def _(r):
    return Spec(
        "Given a dict mapping string keys to ints, return a dict mapping each distinct value (as a string, e.g. 3 -> '3') to the sorted list of keys that have it.",
        "def solve(d):\n    out = {}\n    for k, v in d.items():\n        out.setdefault(str(v), []).append(k)\n    return {v: sorted(ks) for v, ks in out.items()}",
        lambda r_: [_strdict(r_, 7, 0, 4)],
        edges=([{}],),
    )


@template("dicts", "top_k")
def _(r):
    k = r.randint(1, 3)
    return Spec(
        f"Given a dict mapping string keys to ints, return the {k} keys with the largest values as a list, ordered by value descending then key ascending (fewer if the dict is smaller).",
        f"def solve(d):\n    return [k for k, _ in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:{k}]]",
        lambda r_: [_strdict(r_, 7, 0, 5)],
        edges=([{}],),
    )


# ---------------------------------------------------------------- math
@template("math", "gcd_offset")
def _(r):
    a = r.randint(0, 9)
    return Spec(
        f"Return the gcd of all values x + {a} for x in the list of non-negative ints; return 0 for an empty list.",
        f"def solve(a):\n    import math\n    g = 0\n    for x in a:\n        g = math.gcd(g, x + {a})\n    return g",
        lambda r_: [_ints(r_, 0, 40, 0, 6)],
        edges=([[]],),
    )


@template("math", "count_primes_mod")
def _(r):
    m = r.choice([3, 4, 6])
    k = r.choice([x for x in range(1, m) if math.gcd(x, m) == 1])
    return Spec(
        f"Given lo and hi, count primes p with lo <= p <= hi and p % {m} == {k}.",
        f"def solve(lo, hi):\n    def prime(n):\n        if n < 2:\n            return False\n        i = 2\n        while i * i <= n:\n            if n % i == 0:\n                return False\n            i += 1\n        return True\n    return sum(1 for p in range(lo, hi + 1) if prime(p) and p % {m} == {k})",
        lambda r_: (lambda lo: [lo, lo + r_.randint(0, 60)])(r_.randint(0, 80)),
        edges=([0, 0], [2, 2]),
    )


@template("math", "digit_root_base")
def _(r):
    b = r.randint(3, 9)
    return Spec(
        f"Repeatedly replace n by the sum of its base-{b} digits until n < {b}; return n (n >= 0).",
        f"def solve(n):\n    while n >= {b}:\n        s = 0\n        while n:\n            s += n % {b}\n            n //= {b}\n        n = s\n    return n",
        lambda r_: [r_.randint(0, 5000)],
        edges=([0],),
    )


@template("math", "recurrence")
def _(r):
    a, b, mod = r.randint(1, 5), r.randint(0, 4), r.choice([13, 100, 1009])
    return Spec(
        f"Given s0, s1, n define x0 = s0, x1 = s1 and x_k = ({a}*x_(k-1) + {b}*x_(k-2)) % {mod}. Return x_n (n >= 0).",
        f"def solve(s0, s1, n):\n    x, y = s0, s1\n    if n == 0:\n        return x\n    for _ in range(n - 1):\n        x, y = y, ({a} * y + {b} * x) % {mod}\n    return y",
        lambda r_: [r_.randint(0, 20), r_.randint(0, 20), r_.randint(0, 25)],
        edges=([3, 4, 0], [3, 4, 1]),
    )


@template("math", "divisor_sum_limit")
def _(r):
    cap = r.randint(3, 12)
    return Spec(
        f"Return the sum of the divisors of n (n >= 1) that are at most {cap}.",
        f"def solve(n):\n    return sum(d for d in range(1, min(n, {cap}) + 1) if n % d == 0)",
        lambda r_: [r_.randint(1, 400)],
        edges=([1],),
    )


# ---------------------------------------------------------------- dp
@template("dp", "stairs")
def _(r):
    steps = sorted(r.sample([1, 2, 3, 4, 5], r.randint(2, 3)))
    return Spec(
        f"Count the ways to climb exactly n stairs (n >= 0) using steps from {steps}, order matters, modulo 1000000007.",
        f"def solve(n):\n    steps = {steps}\n    w = [1] + [0] * n\n    for i in range(1, n + 1):\n        w[i] = sum(w[i - s] for s in steps if i - s >= 0) % 1000000007\n    return w[n]",
        lambda r_: [r_.randint(0, 30)],
        edges=([0],),
    )


@template("dp", "lis_gap")
def _(r):
    d = r.randint(1, 4)
    return Spec(
        f"Return the length of the longest subsequence in which each chosen element exceeds the previous chosen one by at least {d}. Empty list gives 0.",
        f"def solve(a):\n    best = [1] * len(a)\n    for j in range(len(a)):\n        for i in range(j):\n            if a[j] - a[i] >= {d}:\n                best[j] = max(best[j], best[i] + 1)\n    return max(best, default=0)",
        lambda r_: [_ints(r_, -5, 15, 0, 10)],
        edges=([[]],),
    )


@template("dp", "coin_min")
def _(r):
    coins = sorted(r.sample([1, 2, 3, 4, 5, 7, 9, 11], r.randint(2, 4)))
    return Spec(
        f"Return the minimum number of coins from denominations {coins} (unlimited supply) summing to amount, or -1 if impossible. Amount 0 needs 0 coins.",
        f"def solve(amount):\n    coins = {coins}\n    INF = float('inf')\n    best = [0] + [INF] * amount\n    for v in range(1, amount + 1):\n        for c in coins:\n            if c <= v and best[v - c] + 1 < best[v]:\n                best[v] = best[v - c] + 1\n    return -1 if best[amount] == INF else best[amount]",
        lambda r_: [r_.randint(0, 40)],
        edges=([0],),
    )


@template("dp", "max_window_sum")
def _(r):
    L = r.randint(1, 4)
    return Spec(
        f"Return the maximum sum of a non-empty contiguous subarray of length at most {L}. Empty list gives 0.",
        f"def solve(a):\n    if not a:\n        return 0\n    best = a[0]\n    for i in range(len(a)):\n        t = 0\n        for j in range(i, min(len(a), i + {L})):\n            t += a[j]\n            best = max(best, t)\n    return best",
        lambda r_: [_ints(r_, -6, 8, 0, 9)],
        edges=([[]],),
    )


@template("dp", "edit_distance")
def _(r):
    c = r.choice([1, 2, 3])
    return Spec(
        f"Return the edit distance between strings a and b where insertion and deletion cost 1 and substituting one character costs {c}.",
        f"def solve(a, b):\n    prev = list(range(len(b) + 1))\n    for i in range(1, len(a) + 1):\n        cur = [i] + [0] * len(b)\n        for j in range(1, len(b) + 1):\n            sub = prev[j - 1] + (0 if a[i - 1] == b[j - 1] else {c})\n            cur[j] = min(sub, prev[j] + 1, cur[j - 1] + 1)\n        prev = cur\n    return prev[len(b)]",
        lambda r_: [_text(r_, "abc", 0, 6), _text(r_, "abc", 0, 6)],
        edges=(["", ""],),
    )


# ---------------------------------------------------------------- parsing (held-out family)
def _expr(r, depth=2):
    if depth == 0 or r.random() < 0.3:
        return str(r.randint(0, 20))
    op = r.choice("+-")
    left, right = _expr(r, depth - 1), _expr(r, depth - 1)
    e = f"{left} {op} {right}"
    return f"({e})" if r.random() < 0.5 else e


@template("parsing", "eval_expr")
def _(r):
    mod = r.choice([7, 50, 1000])
    return Spec(
        f"Evaluate an expression of non-negative integers, '+', '-' and parentheses (left to right, spaces allowed) and return the result modulo {mod}.",
        f"def solve(s):\n    return eval(s) % {mod}",
        lambda r_: [_expr(r_)],
        edges=(["5"],),
    )


@template("parsing", "brackets")
def _(r):
    pairs = r.choice(["()[]", "()<>", "[]{}", "(){}[]"])
    opens = pairs[0::2]
    return Spec(
        f"Return True if the string is balanced considering only the bracket pairs {pairs!r} (other characters ignored).",
        f"def solve(s):\n    pairs = {pairs!r}\n    closer = {{pairs[i + 1]: pairs[i] for i in range(0, len(pairs), 2)}}\n    opens = set(pairs[0::2])\n    st = []\n    for c in s:\n        if c in opens:\n            st.append(c)\n        elif c in closer:\n            if not st or st.pop() != closer[c]:\n                return False\n    return not st",
        lambda r_: [_text(r_, pairs + "ab", 0, 10)],
        edges=([""],),
    )


@template("parsing", "kv_parse")
def _(r):
    s1, s2 = r.choice([(";", "="), (",", ":"), ("|", "=")])
    return Spec(
        f"Parse '{s1}'-separated entries of the form key{s2}int into a dict. Later duplicates override earlier ones; skip entries without '{s2}' or whose value is not an integer.",
        f"def solve(s):\n    d = {{}}\n    for part in s.split({s1!r}):\n        if {s2!r} not in part:\n            continue\n        k, v = part.split({s2!r}, 1)\n        try:\n            d[k] = int(v)\n        except ValueError:\n            pass\n    return d",
        lambda r_: [s1.join(r_.choice(["a", "b", "c", "dd"]) + r_.choice([s2, s2, ""]) + r_.choice(["1", "22", "-3", "x", ""]) for _ in range(r_.randint(0, 6)))],
        edges=([""],),
    )


@template("parsing", "word_freq")
def _(r):
    n = r.randint(1, 3)
    return Spec(
        f"Lower-case the text, split into words of ASCII letters (any other character separates words), keep words of length at least {n}, and return [word, count] pairs sorted by count descending then word ascending.",
        f"def solve(s):\n    import re\n    c = {{}}\n    for w in re.findall('[a-z]+', s.lower()):\n        if len(w) >= {n}:\n            c[w] = c.get(w, 0) + 1\n    return [[w, c[w]] for w in sorted(c, key=lambda w: (-c[w], w))]",
        lambda r_: [" ".join(r_.choice(["the", "Cat", "a", "dog,", "cat.", "IS", "x1y"]) for _ in range(r_.randint(0, 8)))],
        edges=([""],),
    )


@template("parsing", "csv_column_sum")
def _(r):
    col = r.randint(0, 2)
    return Spec(
        f"The input has lines of comma-separated fields. Return the sum of field {col} (0-based) over lines that have that field and where it parses as an integer.",
        f"def solve(s):\n    t = 0\n    for line in s.split('\\n'):\n        f = line.split(',')\n        if len(f) > {col}:\n            try:\n                t += int(f[{col}])\n            except ValueError:\n                pass\n    return t",
        lambda r_: ["\n".join(",".join(r_.choice(["1", "20", "-4", "x", ""]) for _ in range(r_.randint(1, 4))) for _ in range(r_.randint(0, 5)))],
        edges=([""],),
    )


# ---------------------------------------------------------------- building
def _json_norm(x: Any) -> Any:
    return json.loads(json.dumps(x, allow_nan=False))


def build_task(family: str, name: str, index: int, seed: int, split: Split) -> Task:
    """Deterministic task for (family, template, index, seed)."""
    r = random.Random(f"task:{seed}:{family}:{name}:{index}")
    spec = _TEMPLATES[family][name](r)
    ns: dict = {}
    exec(compile(spec.reference, f"<ref {family}/{name}>", "exec"), ns)  # trusted reference
    ref = ns[ENTRY]
    seen: set[str] = set()
    tests: list[CheckCase] = []
    gen_rng = random.Random(f"tests:{seed}:{family}:{name}:{index}")

    def add(args: list) -> bool:
        key = json.dumps(args, sort_keys=True)
        if key in seen:
            return False
        seen.add(key)
        tests.append(CheckCase(tuple(_json_norm(args)), _json_norm(ref(*_json_norm(args)))))
        return True

    # visible examples first: prefer non-edge inputs so the prompt shows typical cases
    attempts = 0
    while len(tests) < N_VISIBLE and attempts < 200:
        add(spec.gen(gen_rng))
        attempts += 1
    for e in spec.edges:
        if len(tests) < N_TESTS:
            add(list(e))
    while len(tests) < N_TESTS and attempts < 2000:
        add(spec.gen(gen_rng))
        attempts += 1
    if len(tests) < N_TESTS:
        raise RuntimeError(f"could not generate {N_TESTS} distinct tests for {family}/{name}")
    examples = "\n".join(f"  {ENTRY}({', '.join(json.dumps(a) for a in t.args)}) == {json.dumps(t.expected)}" for t in tests[:N_VISIBLE])
    prompt = f"Write a Python function {ENTRY} that does the following.\n{spec.description}\nExamples:\n{examples}\n"
    return Task(f"{split.value}/{family}/{name}/{index:04d}", family, name, split, prompt, spec.reference, tuple(tests))


def templates_of(family: str) -> list[str]:
    return sorted(_TEMPLATES[family])


def assign_splits(seed: int) -> dict[tuple[str, str], Split]:
    """Template-level split. Every stream family: 3 train, 1 dev, 1 test. Held-out family: 2 dev, 3 test."""
    out: dict[tuple[str, str], Split] = {}
    for fam in STREAM_FAMILIES:
        names = templates_of(fam)
        random.Random(f"split:{seed}:{fam}").shuffle(names)
        roles = [Split.TRAIN] * 3 + [Split.DEV, Split.FINAL_TEST]
        out.update({(fam, n): s for n, s in zip(names, roles)})
    for fam in HELDOUT_FAMILIES:
        names = templates_of(fam)
        random.Random(f"split:{seed}:{fam}").shuffle(names)
        roles = [Split.DEV] * 2 + [Split.FINAL_TEST] * 3
        out.update({(fam, n): s for n, s in zip(names, roles)})
    return out


class TaskBank:
    """All tasks for a seed, grouped by split. Instances are generated lazily and deterministically."""

    def __init__(self, seed: int):
        self.seed = seed
        self.splits = assign_splits(seed)

    def templates(self, split: Split, family: Optional[str] = None) -> list[tuple[str, str]]:
        return sorted(k for k, s in self.splits.items() if s is split and (family is None or k[0] == family))

    def tasks(self, split: Split, family: str, n_per_template: int, offset: int = 0) -> list[Task]:
        out = []
        for fam, name in self.templates(split, family):
            for i in range(offset, offset + n_per_template):
                out.append(build_task(fam, name, i, self.seed, split))
        return out
