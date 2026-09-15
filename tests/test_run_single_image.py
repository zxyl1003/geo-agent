from geoagent.core.schemas import VisualCue
from geoagent.state.task_state import GeoLocalizationState
from scripts.run_single_image import print_state_summary


def test_state_summary_handles_visual_cue_without_confidence(capsys):
    state = GeoLocalizationState(
        image_path="demo.jpg",
        visual_cues=[VisualCue(cue_type="scene", text="urban street", confidence=None)],
    )

    print_state_summary(state)

    assert "[scene] urban street (unknown)" in capsys.readouterr().out
