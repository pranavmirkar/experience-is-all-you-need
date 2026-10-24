"""Vintage-2 templates: four per stream family and two for the held-out family.

Written after the dev diagnostic of the vintage-1 bank, but before any model was run on these
templates. I did see which vintage-1 dev templates the model failed (multi-condition specs such as
merge_filter, digit_root_base and kv_parse), which could bias me toward easier specs here; this is
recorded in the protocol. The templates are meant to cover a spread from one-rule to
multi-condition specs and were not tuned against any model output.
"""

from __future__ import annotations

from .tasks import Spec, _ints, _strdict, _text, template

V = 2


# ---------------------------------------------------------------- strings
@template("strings", "count_overlap", V)
def _(r):
    p = r.choice(["aa", "ab", "aba", "bb", "aab"])
    return Spec(
        f"Count how many times {p!r} occurs in the string, counting overlapping occurrences.",
        f"def solve(s):\n    return sum(1 for i in range(len(s) - {len(p)} + 1) if s.startswith({p!r}, i))",
        lambda r_: [_text(r_, "ab", 3, 16)],
        edges=([""],),
    )


@template("strings", "case_by_index", V)
def _(r):
    m = r.randint(2, 4)
    return Spec(
        f"Upper-case the characters at indices divisible by {m} and lower-case all other characters.",
        f"def solve(s):\n    return ''.join(c.upper() if i % {m} == 0 else c.lower() for i, c in enumerate(s))",
        lambda r_: [_text(r_, "abcXYZ -", 0, 14)],
        edges=([""],),
    )


@template("strings", "limit_repeats", V)
def _(r):
    k = r.randint(1, 3)
    return Spec(
        f"Shorten every run of equal consecutive characters to at most {k} characters.",
        f"def solve(s):\n    out = []\n    for c in s:\n        if len(out) >= {k} and all(x == c for x in out[-{k}:]):\n            continue\n        out.append(c)\n    return ''.join(out)",
        lambda r_: [_text(r_, "aab", 0, 16)],
        edges=([""], ["aaaaa"]),
    )


@template("strings", "digit_cipher", V)
def _(r):
    m, c = r.choice([(3, 1), (7, 2), (3, 4), (9, 5)])
    return Spec(
        f"Replace every decimal digit d by (d * {m} + {c}) % 10. Leave all other characters unchanged.",
        f"def solve(s):\n    return ''.join(str((int(ch) * {m} + {c}) % 10) if ch in '0123456789' else ch for ch in s)",
        lambda r_: [_text(r_, "ab 0123456789", 0, 14)],
        edges=([""],),
    )


# ---------------------------------------------------------------- lists
@template("lists", "window_sums", V)
def _(r):
    w = r.randint(2, 4)
    return Spec(
        f"Return the list of sums of every {w} consecutive elements, left to right. Return an empty list if there are fewer than {w} elements.",
        f"def solve(a):\n    return [sum(a[i:i + {w}]) for i in range(len(a) - {w} + 1)]",
        lambda r_: [_ints(r_, -5, 9, 0, 10)],
        edges=([[]],),
    )


@template("lists", "unique_keep_last", V)
def _(r):
    return Spec(
        "Remove duplicate values, keeping only the last occurrence of each value. The kept elements stay in their original relative order.",
        "def solve(a):\n    last = {}\n    for i, x in enumerate(a):\n        last[x] = i\n    return [x for i, x in enumerate(a) if last[x] == i]",
        lambda r_: [_ints(r_, 0, 5, 0, 12)],
        edges=([[]],),
    )


@template("lists", "chunk_sums", V)
def _(r):
    c = r.randint(2, 4)
    return Spec(
        f"Split the list into consecutive chunks of {c} elements (the last chunk may be shorter) and return the list of chunk sums.",
        f"def solve(a):\n    return [sum(a[i:i + {c}]) for i in range(0, len(a), {c})]",
        lambda r_: [_ints(r_, -5, 9, 0, 11)],
        edges=([[]],),
    )


@template("lists", "best_rise", V)
def _(r):
    return Spec(
        "Return the largest value of a[j] - a[i] over all pairs of indices i < j. Return 0 if there is no such pair or the largest difference is negative.",
        "def solve(a):\n    best = 0\n    lo = None\n    for x in a:\n        if lo is not None:\n            best = max(best, x - lo)\n        lo = x if lo is None else min(lo, x)\n    return best",
        lambda r_: [_ints(r_, -9, 20, 0, 10)],
        edges=([[]], [[5]]),
    )


# ---------------------------------------------------------------- dicts
@template("dicts", "initial_counts", V)
def _(r):
    t = r.randint(1, 3)
    return Spec(
        f"The input is a list of lowercase words. Return a dict mapping each first letter to how many words start with it, counting only words of length at least {t}.",
        f"def solve(words):\n    d = {{}}\n    for w in words:\n        if len(w) >= {t}:\n            d[w[0]] = d.get(w[0], 0) + 1\n    return d",
        lambda r_: [[_text(r_, "abcd", 1, 4) for _ in range(r_.randint(0, 8))]],
        edges=([[]],),
    )


@template("dicts", "max_per_key", V)
def _(r):
    t = r.randint(-2, 4)
    return Spec(
        f"The input is a dict mapping string keys to lists of ints. Return a dict mapping each key to the maximum of its list, leaving out keys with an empty list and keys whose maximum is below {t}.",
        f"def solve(d):\n    return {{k: max(v) for k, v in d.items() if v and max(v) >= {t}}}",
        lambda r_: [{_text(r_, "abcde", 1, 2): _ints(r_, -4, 8, 0, 4) for _ in range(r_.randint(0, 5))}],
        edges=([{}],),
    )


@template("dicts", "anagram_groups", V)
def _(r):
    return Spec(
        "The input is a list of lowercase words. Group words that are anagrams of each other. Return a list of the groups that contain at least 2 words (duplicates count), each group sorted ascending, and the list of groups sorted ascending.",
        "def solve(words):\n    g = {}\n    for w in words:\n        g.setdefault(''.join(sorted(w)), []).append(w)\n    return sorted(sorted(v) for v in g.values() if len(v) >= 2)",
        lambda r_: [[_text(r_, "abc", 1, 3) for _ in range(r_.randint(3, 9))]],
        edges=([[]], [["ab", "ba"]]),
    )


@template("dicts", "increment_prune", V)
def _(r):
    s = r.randint(1, 3)
    return Spec(
        f"Given a dict d of string keys to ints and a list of keys, add {s} to d[key] for every occurrence of key in the list (a missing key starts at 0). Return the resulting dict without any entry whose value is 0 or below.",
        f"def solve(d, keys):\n    out = dict(d)\n    for k in keys:\n        out[k] = out.get(k, 0) + {s}\n    return {{k: v for k, v in out.items() if v > 0}}",
        lambda r_: [_strdict(r_, 5, -4, 4), [_text(r_, "abcdef", 1, 2) for _ in range(r_.randint(0, 6))]],
        edges=([{}, []],),
    )


# ---------------------------------------------------------------- math
@template("math", "digit_sum_multiples", V)
def _(r):
    m = r.randint(2, 9)
    return Spec(
        f"Return the sum of the decimal digits of all multiples of {m} in the range 1..n inclusive. Return 0 if n < {m}.",
        f"def solve(n):\n    return sum(sum(int(c) for c in str(x)) for x in range({m}, n + 1, {m}))",
        lambda r_: [r_.randint(0, 150)],
        edges=([0],),
    )


@template("math", "collatz_stat", V)
def _(r):
    stat = r.choice(["steps", "peak"])
    desc = (
        "the number of steps needed to reach 1" if stat == "steps" else "the largest value reached (including n itself)"
    )
    body = (
        "    steps = 0\n    while n != 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        steps += 1\n    return steps"
        if stat == "steps"
        else "    peak = n\n    while n != 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        peak = max(peak, n)\n    return peak"
    )
    return Spec(
        f"Starting from n >= 1, repeatedly replace n by n // 2 if n is even, else by 3 * n + 1, until n is 1. Return {desc}.",
        f"def solve(n):\n{body}",
        lambda r_: [r_.randint(1, 80)],
        edges=([1],),
    )


@template("math", "base_palindromes", V)
def _(r):
    b = r.choice([2, 3, 4, 5])
    return Spec(
        f"Count the integers x with lo <= x <= hi whose representation in base {b} (no leading zeros; 0 is '0') reads the same forwards and backwards. Return 0 if lo > hi.",
        f"def solve(lo, hi):\n    def digits(x):\n        if x == 0:\n            return [0]\n        out = []\n        while x:\n            out.append(x % {b})\n            x //= {b}\n        return out\n    return sum(1 for x in range(lo, hi + 1) if digits(x) == digits(x)[::-1])",
        lambda r_: (lambda lo: [lo, lo + r_.randint(0, 60)])(r_.randint(0, 60)),
        edges=([5, 2],),
    )


@template("math", "fib_prefix_mod", V)
def _(r):
    m = r.choice([7, 10, 100, 1000])
    return Spec(
        f"Return the sum of the first n Fibonacci numbers (1, 1, 2, 3, 5, ...) modulo {m}. For n = 0 the sum is 0.",
        f"def solve(n):\n    a, b, t = 1, 1, 0\n    for _ in range(n):\n        t += a\n        a, b = b, a + b\n    return t % {m}",
        lambda r_: [r_.randint(0, 45)],
        edges=([0],),
    )


# ---------------------------------------------------------------- dp
@template("dp", "grid_paths", V)
def _(r):
    return Spec(
        "The input is a list of equal-length strings made of '.' (open) and '#' (blocked). Count the paths from the top-left to the bottom-right cell that move only right or down and never enter a blocked cell. If either end cell is blocked, or the grid is empty, the count is 0.",
        "def solve(grid):\n    if not grid or not grid[0] or grid[0][0] == '#' or grid[-1][-1] == '#':\n        return 0\n    h, w = len(grid), len(grid[0])\n    ways = [[0] * w for _ in range(h)]\n    ways[0][0] = 1\n    for i in range(h):\n        for j in range(w):\n            if grid[i][j] == '#' or (i == 0 and j == 0):\n                continue\n            ways[i][j] = (ways[i - 1][j] if i else 0) + (ways[i][j - 1] if j else 0)\n    return ways[-1][-1]",
        lambda r_: [(lambda h, w: ["".join(r_.choice(".....#") for _ in range(w)) for _ in range(h)])(r_.randint(1, 4), r_.randint(1, 4))],
        edges=([[]], [["."]], [["#"]]),
    )


@template("dp", "spaced_pick_max", V)
def _(r):
    g = r.randint(1, 2)
    return Spec(
        f"Choose elements of the list of non-negative ints so that any two chosen indices differ by more than {g}. Return the maximum possible sum (0 for an empty list).",
        f"def solve(a):\n    best = []\n    for i, x in enumerate(a):\n        take = x + (best[i - {g} - 1] if i - {g} - 1 >= 0 else 0)\n        skip = best[i - 1] if i else 0\n        best.append(max(take, skip))\n    return best[-1] if best else 0",
        lambda r_: [_ints(r_, 0, 9, 0, 9)],
        edges=([[]],),
    )


@template("dp", "subset_sum_count", V)
def _(r):
    return Spec(
        "Count the subsets of indices of the list a (all positive ints) whose elements sum to exactly t. The empty subset counts when t is 0.",
        "def solve(a, t):\n    ways = [1] + [0] * max(t, 0)\n    if t < 0:\n        return 0\n    for x in a:\n        for s in range(t, x - 1, -1):\n            ways[s] += ways[s - x]\n    return ways[t]",
        lambda r_: [_ints(r_, 1, 6, 0, 8), r_.randint(0, 14)],
        edges=([[], 0], [[], 3]),
    )


@template("dp", "common_run", V)
def _(r):
    kind = r.choice(["substring", "subsequence"])
    if kind == "substring":
        desc = "Return the length of the longest contiguous substring that occurs in both strings a and b."
        body = "def solve(a, b):\n    best = 0\n    prev = [0] * (len(b) + 1)\n    for i in range(1, len(a) + 1):\n        cur = [0] * (len(b) + 1)\n        for j in range(1, len(b) + 1):\n            if a[i - 1] == b[j - 1]:\n                cur[j] = prev[j - 1] + 1\n                best = max(best, cur[j])\n        prev = cur\n    return best"
    else:
        desc = "Return the length of the longest common subsequence of strings a and b."
        body = "def solve(a, b):\n    prev = [0] * (len(b) + 1)\n    for i in range(1, len(a) + 1):\n        cur = [0] * (len(b) + 1)\n        for j in range(1, len(b) + 1):\n            cur[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], cur[j - 1])\n        prev = cur\n    return prev[len(b)]"
    return Spec(
        desc,
        body,
        lambda r_: [_text(r_, "abc", 0, 8), _text(r_, "abc", 0, 8)],
        edges=(["", ""], ["abc", ""]),
    )


# ---------------------------------------------------------------- parsing (held-out family)
@template("parsing", "version_compare", V)
def _(r):
    return Spec(
        "Compare two dotted version strings of non-negative ints. A missing trailing part counts as 0. Return -1 if a < b, 0 if equal, 1 if a > b.",
        "def solve(a, b):\n    x = [int(p) for p in a.split('.')]\n    y = [int(p) for p in b.split('.')]\n    n = max(len(x), len(y))\n    x += [0] * (n - len(x))\n    y += [0] * (n - len(y))\n    return (x > y) - (x < y)",
        lambda r_: [".".join(str(r_.randint(0, 12)) for _ in range(r_.randint(1, 3))) for _ in range(2)],
        edges=(["1.0", "1"], ["2", "10"]),
    )


@template("parsing", "duration_parse", V)
def _(r):
    unit, div = r.choice([("seconds", 1), ("minutes", 60)])
    return Spec(
        f"Parse a duration made of number+unit tokens, where the unit is h, m or s, each unit appears at most once and the tokens may come in any order (for example '1h30s' or '5m2h'). Return the total in whole {unit} (rounded down). The empty string is 0.",
        f"def solve(s):\n    import re\n    mult = {{'h': 3600, 'm': 60, 's': 1}}\n    total = sum(int(n) * mult[u] for n, u in re.findall(r'(\\d+)([hms])', s))\n    return total // {div}",
        lambda r_: ["".join(f"{r_.randint(0, 99)}{u}" for u in r_.sample("hms", r_.randint(1, 3)))],
        edges=([""],),
    )
