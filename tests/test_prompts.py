from geoagent.core.prompt_loader import load_prompt


def test_brain_prompt_loads_shared_sections_and_current_decision_contract():
    prompt = load_prompt(
        "configs",
        "brain_agent.md",
        shared_names=("coordinate_contract.md", "provider_attribution.md", "granularity_levels.md"),
    )

    assert "Shared Coordinate Contract" in prompt
    assert "Shared Provider Attribution Rules" in prompt
    assert "Shared Granularity Levels" in prompt
    assert "GeoLocalization Brain Agent" in prompt
    assert "`call_tools`" in prompt
    assert '"tool_requests"' in prompt
    assert '"memory_references"' in prompt
    assert '"visual_analysis"' in prompt
    assert "On later decisions, the source image is not attached" in prompt
    assert "requesting `visual_reanalysis` only" in prompt
    assert "never set one of these child fields to `null`" in prompt
    assert 'an add-only patch must use `"update": []` and `"remove": []`' in prompt
    assert '"visual_updates": null' in prompt
    assert "Duplicate tool call with same arguments." in prompt
    assert "at most three concise sentences" in prompt
    assert "`no_op`" not in prompt
    assert '"tool_request"' not in prompt


def test_brain_perception_prompt_contains_schema_and_cue_types():
    prompt = load_prompt("configs", "brain_perception.md")

    assert "Cue Type Hierarchy" in prompt
    assert '"visual_cues"' in prompt
    assert "`road_marking`" in prompt
    assert '"entity_type": "poi|landmark|phone|address|sign|vehicle|other"' in prompt
    assert "directly recognizable landmarks" in prompt
    assert "non-text regional clues" in prompt
    assert "not an intermediate input" in prompt


def test_memory_manager_prompt_forbids_place_facts_and_direct_mutation():
    prompt = load_prompt("configs", "memory_manager_agent.md")
    compact = " ".join(prompt.split())

    assert "never write to the database directly" in compact
    assert "not a place-fact knowledge entry" in compact
    assert '"candidate"' in prompt


def test_brain_prompt_owns_confidence_and_requires_final_coordinates():
    prompt = load_prompt(
        "configs",
        "brain_agent.md",
        shared_names=("coordinate_contract.md", "provider_attribution.md", "granularity_levels.md"),
    )

    assert "street reference point" in prompt
    assert "do not include proposed future tool calls" in prompt
    assert "full house-number address and lat/lon" in prompt
    assert "Every international POI search must include `country` as an ISO 3166-1" in prompt
    assert "Never pass provider-specific `gl`, `hl`, `location`, or `map_provider`" in prompt
    assert "confidence" in prompt
    assert "uncertainty_radius_m" in prompt
    assert "Always include coordinates" in prompt
    assert "EvidenceReview" not in prompt
    assert "memory.rejected_candidates" not in prompt
