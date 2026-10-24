from experience_learning.schemas import Episode, Split, ToolOutput, Verification


def make_episode(
    eid="e1",
    family="A",
    split=Split.TRAIN,
    verified=True,
    tokens=100,
    loss=0.5,
    embedding=(1.0, 0.0),
    stream_index=0,
):
    ver = (
        Verification(True, "unit_tests", "pytest -q", True)
        if verified
        else Verification(False, "model_assertion")
    )
    return Episode(
        episode_id=eid,
        created_at="2026-01-01T00:00:00+00:00",
        stream_index=stream_index,
        family=family,
        language="python",
        source="test",
        split=split,
        prompt="p",
        attempt="a",
        tool_outputs=(ToolOutput("test", False, "fail"),),
        correction="c",
        verification=ver,
        model_checkpoint="base",
        adapter_version="none",
        tokens=tokens,
        wall_time_s=0.1,
        license="CC0",
        loss=loss,
        embedding=embedding,
    )
