import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from experience_learning.schemas import Split  # noqa: E402
from experience_learning.stage2 import tasks as T  # noqa: E402

# stage2_smoke imports only the stdlib and repo code at module level (torch is imported inside main).
import stage2_smoke as S  # noqa: E402


def test_shots_are_correct_and_not_bank_templates():
    for desc, examples, code in S.SHOTS:
        ns: dict = {}
        exec(code, ns)
        for e in examples:
            lhs, rhs = e.split(" == ")
            assert eval(lhs, ns) == eval(rhs)
        for fam in T.STREAM_FAMILIES + T.HELDOUT_FAMILIES:
            for name in T.templates_of(fam):
                assert T.build_task(fam, name, 0, 0, Split.DEV).prompt.split("\n")[1] != desc


def test_fewshot_prompt_keeps_the_zero_shot_suffix():
    t = T.build_task("strings", "vowel_positions", 0, 0, Split.DEV)
    zero, few = S.format_prompt(t), S.format_prompt(t, fewshot=True)
    assert few.endswith(zero) and len(few) > len(zero) and few.count("def solve(") == 3
    assert t.reference_source not in few
