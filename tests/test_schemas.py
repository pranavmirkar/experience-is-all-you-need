import pytest

from experience_learning.schemas import Episode, SchemaError, Split, Verification

from helpers import make_episode


def test_roundtrip():
    e = make_episode()
    assert Episode.from_dict(e.to_dict()) == e


def test_model_assertion_cannot_be_verified():
    with pytest.raises(SchemaError):
        Verification(True, "model_assertion", "trust me", True)


def test_verified_requires_independence_and_procedure():
    with pytest.raises(SchemaError):
        Verification(True, "unit_tests", "pytest", False)
    with pytest.raises(SchemaError):
        Verification(True, "unit_tests", "", True)


def test_invalid_fields_rejected():
    with pytest.raises(SchemaError):
        make_episode(tokens=0)
    with pytest.raises(SchemaError):
        make_episode(eid="")


def test_split_is_immutable():
    e = make_episode()
    with pytest.raises(Exception):
        e.split = Split.DEV  # type: ignore[misc]
