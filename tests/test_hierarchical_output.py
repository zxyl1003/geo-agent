import pytest
from pydantic import ValidationError

from geoagent.core.schemas import BrainDecision, FinalAnswer, Hypothesis, ToolRequest


def test_failure_record_can_omit_coordinates():
    answer = FinalAnswer(location_name="Unknown", granularity="unknown", confidence=0.2)

    assert answer.lat is None
    assert answer.lon is None


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(35.0, None), (None, 139.0), (91.0, 0.0), (0.0, 181.0)],
)
def test_final_answer_rejects_invalid_coordinate_contract(lat, lon):
    with pytest.raises(ValidationError):
        FinalAnswer(
            location_name="Invalid",
            lat=lat,
            lon=lon,
            granularity="coordinates",
            confidence=0.5,
        )


def test_final_answer_rejects_coordinates_with_unknown_granularity():
    with pytest.raises(ValidationError):
        FinalAnswer(location_name="Invalid", lat=35.0, lon=139.0, granularity="unknown", confidence=0.5)


def test_hypothesis_coordinates_are_range_checked():
    with pytest.raises(ValidationError):
        Hypothesis(name="Invalid", lat=-91.0, lon=0.0)


def test_final_decision_requires_matching_brain_confidence():
    with pytest.raises(ValidationError):
        BrainDecision(
            action_type="final_answer",
            confidence=0.8,
            final_answer=FinalAnswer(
                location_name="Tokyo",
                lat=35.0,
                lon=139.0,
                granularity="city",
                confidence=0.7,
            ),
        )


def test_tool_action_cardinality_is_structural():
    with pytest.raises(ValidationError):
        BrainDecision(action_type="call_tool", tool_requests=[], confidence=0.2)

    with pytest.raises(ValidationError):
        BrainDecision(
            action_type="call_tools",
            tool_requests=[ToolRequest(tool_name="ocr")],
            confidence=0.2,
        )
