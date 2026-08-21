import pytest

from fusion_runtime.completion import (
    FINAL_ANSWER_MISSING,
    INVALID_TOOL_CALL,
    OUTPUT_TRUNCATED,
    PROVIDER_PROTOCOL_ERROR,
    CompletionTracker,
    classify_response,
    normalize_finish_reason,
)
from fusion_runtime.types import (
    CompletionRecord,
    Finish,
    ModelResponse,
    StreamError,
    TextDelta,
    ToolCallDelta,
    Usage,
)


def _tool_call(arguments: str = '{"path":"README.md"}') -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": arguments},
    }


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("stop", "stop"),
        ("end_turn", "stop"),
        ("stop-sequence", "stop"),
        ("tool_use", "tool_calls"),
        ("function call", "tool_calls"),
        ("max_tokens", "length"),
        ("", None),
        (None, None),
    ],
)
def test_finish_reason_normalization(reason, expected):
    assert normalize_finish_reason(reason) == expected


@pytest.mark.parametrize(
    ("response", "events"),
    [
        (
            ModelResponse(
                content="done",
                finish_reason="end_turn",
                usage={"total_tokens": 7},
            ),
            [TextDelta("do"), TextDelta("ne"), Finish("stop"), Usage({"total_tokens": 7})],
        ),
        (
            ModelResponse(
                content="",
                tool_calls=[_tool_call()],
                finish_reason="tool_calls",
                usage={"total_tokens": 7},
            ),
            [
                ToolCallDelta(index=0, id="call_1", name="read_file"),
                ToolCallDelta(index=0, arguments='{"path":'),
                ToolCallDelta(index=0, arguments='"README.md"}'),
                Finish("tool_use"),
                Usage({"total_tokens": 7}),
            ],
        ),
        (
            ModelResponse(content="", finish_reason="max_tokens"),
            [Finish("length")],
        ),
        (
            ModelResponse(
                content="",
                tool_calls=[_tool_call("{broken")],
                finish_reason="tool_calls",
            ),
            [
                ToolCallDelta(index=0, id="call_1", name="read_file", arguments="{broken"),
                Finish("tool_calls"),
            ],
        ),
    ],
)
def test_complete_and_stream_use_identical_completion_classification(response, events):
    record = CompletionRecord()
    tracker = CompletionTracker(record)
    for event in events:
        tracker.observe(event)

    assert record.outcome == classify_response(response)


def test_empty_truncated_response_has_stable_failure_tag_order():
    outcome = classify_response(ModelResponse(content="   ", finish_reason="max_output_tokens"))

    assert outcome.status == "incomplete"
    assert outcome.finish_reason == "length"
    assert outcome.has_public_output is False
    assert outcome.usage_reported is False
    assert outcome.failure_tags == (FINAL_ANSWER_MISSING, OUTPUT_TRUNCATED)


def test_invalid_tool_call_is_not_treated_as_a_valid_public_answer():
    outcome = classify_response(
        ModelResponse(content="", tool_calls=[_tool_call("not-json")], finish_reason="tool_calls")
    )

    assert outcome.status == "incomplete"
    assert outcome.tool_call_count == 1
    assert outcome.has_tool_call is True
    assert outcome.has_valid_tool_call is False
    assert outcome.has_public_output is False
    assert outcome.failure_tags == (FINAL_ANSWER_MISSING, INVALID_TOOL_CALL)


def test_stream_tracker_exposes_pending_then_final_immutable_snapshots():
    record = CompletionRecord()
    tracker = CompletionTracker(record)
    initial = record.outcome

    tracker.observe(TextDelta("answer"))
    pending = record.outcome
    tracker.observe(Finish("stop"))
    finished = record.outcome
    tracker.observe(Usage({"total_tokens": 4}))

    assert initial.status == "pending"
    assert initial.has_public_output is False
    assert pending.status == "pending"
    assert pending.has_public_output is True
    assert finished.status == "completed"
    assert finished.usage_reported is False
    assert record.outcome.status == "completed"
    assert record.outcome.usage_reported is True


def test_stream_ending_without_terminal_is_an_infrastructure_failure():
    record = CompletionRecord()
    tracker = CompletionTracker(record)

    tracker.end()

    assert record.outcome.status == "failed"
    assert record.outcome.infrastructure_failure is True
    assert record.outcome.failure_tags == (FINAL_ANSWER_MISSING, PROVIDER_PROTOCOL_ERROR)


def test_protocol_error_after_public_text_preserves_output_evidence():
    record = CompletionRecord()
    tracker = CompletionTracker(record)
    tracker.observe(TextDelta("partial"))

    tracker.observe(StreamError("bad stream", code="provider_protocol_error"))

    assert record.outcome.status == "failed"
    assert record.outcome.has_public_output is True
    assert record.outcome.infrastructure_failure is True
    assert record.outcome.failure_tags == (PROVIDER_PROTOCOL_ERROR,)


@pytest.mark.parametrize("with_partial_text", [False, True])
def test_consumer_cancellation_is_not_mislabeled_as_model_failure(with_partial_text):
    record = CompletionRecord()
    tracker = CompletionTracker(record)
    if with_partial_text:
        tracker.observe(TextDelta("partial"))

    tracker.cancel()

    assert record.outcome.status == "cancelled"
    assert record.outcome.has_public_output is with_partial_text
    assert record.outcome.infrastructure_failure is False
    assert record.outcome.failure_tags == ()
