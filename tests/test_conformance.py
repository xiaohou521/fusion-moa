import pytest

from fusion_runtime.conformance import (
    StreamContractError,
    assert_stream_conforms,
    collect_stream,
    stream_violations,
)
from fusion_runtime.types import Finish, StreamError, TextDelta, ToolCallDelta, Usage


def test_complete_canonical_stream_conforms():
    events = [
        TextDelta("hello"),
        ToolCallDelta(index=0, id="call_1", name="read_file"),
        ToolCallDelta(index=0, arguments='{"path":"README.md"}'),
        Finish("tool_calls"),
        Usage({"input_tokens": 2, "output_tokens": 3}),
    ]

    assert_stream_conforms(events)
    assert stream_violations(events) == []


def test_terminal_stream_error_conforms():
    assert_stream_conforms(
        [
            TextDelta("partial"),
            StreamError("upstream provider stream failed", retryable=True),
        ]
    )


def test_conformance_reports_all_ordering_and_value_errors():
    events = [
        TextDelta(""),
        ToolCallDelta(index=-1),
        Usage({"tokens": -1}),
        Finish(""),
        TextDelta("after finish"),
        StreamError("", code=""),
    ]

    with pytest.raises(StreamContractError) as captured:
        assert_stream_conforms(events)

    violations = captured.value.violations
    assert "stream cannot contain both Finish and StreamError" in violations
    assert "only Usage may follow Finish" in violations
    assert "TextDelta.text must not be empty" in violations
    assert "ToolCallDelta.index must be non-negative" in violations
    assert "ToolCallDelta must carry id, name, or arguments" in violations
    assert "Usage values must be non-negative integers" in violations
    assert "Finish.reason must not be empty" in violations
    assert "StreamError.message must not be empty" in violations
    assert "StreamError.code must not be empty" in violations


async def test_collect_stream_supports_plugin_conformance_tests():
    async def events():
        yield TextDelta("ok")
        yield Finish("stop")

    assert await collect_stream(events()) == [TextDelta("ok"), Finish("stop")]
