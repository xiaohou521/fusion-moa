import pytest

from fusion_runtime.accounting import (
    ATTEMPT_USAGE_MISSING,
    USAGE_MISSING,
    USAGE_TOKENS_MISSING,
    USAGE_TOTAL_MISMATCH,
    USAGE_VALUE_INVALID,
    assess_usage,
)


@pytest.mark.parametrize(
    "usage",
    [
        {"total_tokens": 7},
        {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        {"input_tokens": 4, "output_tokens": 3},
        {
            "input_tokens": 4,
            "output_tokens": 3,
            "total_tokens": 7,
            "input_tokens_details": {"cached_tokens": 2},
        },
        {
            "prompt_tokens": 4,
            "completion_tokens": 3,
            "total_tokens": 7,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "multimodal_tokens": None,
            },
        },
    ],
)
def test_common_provider_usage_shapes_are_complete(usage):
    reported, issues = assess_usage(usage, report_seen=True)

    assert reported is True
    assert issues == ()


def test_absent_usage_is_not_interpreted_as_zero_cost():
    reported, issues = assess_usage({}, report_seen=False)

    assert reported is False
    assert issues == (USAGE_MISSING,)


def test_known_usage_from_only_some_attempts_is_incomplete():
    reported, issues = assess_usage(
        {"total_tokens": 7},
        report_seen=True,
        reported_for_all_attempts=False,
    )

    assert reported is False
    assert issues == (ATTEMPT_USAGE_MISSING,)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"latency_ms": 12}, (USAGE_TOKENS_MISSING,)),
        ({"total_tokens": -1}, (USAGE_VALUE_INVALID,)),
        ({"total_tokens": True}, (USAGE_VALUE_INVALID,)),
        ({"total_tokens": None}, (USAGE_VALUE_INVALID,)),
        (
            {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 99},
            (USAGE_TOTAL_MISMATCH,),
        ),
    ],
)
def test_invalid_or_contradictory_usage_has_stable_issue_codes(usage, expected):
    reported, issues = assess_usage(usage, report_seen=True)

    assert reported is True
    assert issues == expected
