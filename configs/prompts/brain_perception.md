# Direct Visual Observation

When the source image is attached on the initial Brain decision, inspect it directly and record the visible evidence thoroughly. Later Brain decisions do not receive the source image; they continue from the recorded visual evidence and tool results, and may request `visual_reanalysis` for a concrete targeted inspection when necessary.

On the initial Brain decision, record the observable evidence in the top-level `visual_analysis` field while also selecting the next geolocation action. This field is an auditable account of what you observed, not an intermediate input that limits your reasoning. On every later decision, set both `visual_analysis` and `visual_updates` to null. Evidence returned by `visual_reanalysis` is merged into the compact state automatically; do not copy or restate it as a visual update. Be thorough and specific on the initial inspection. Generic observations like "urban street" are weak unless no better clue is visible.

Street-view watermarks such as `{year} Google`, `{year} Baidu`, `@ {year} Google`, or `@ {year} Baidu` are capture-provider overlays, not useful text evidence; do not return them.

## Entity Attribution

When the image contains multiple storefronts, signs, POIs, phone numbers, or addresses, keep their text grouped by visible object or storefront. Do not attach a phone number to a POI unless it is visually on the same sign, facade, window, poster, or storefront area.

Use `observed_entities` to represent these groups. If a phone number is visible but its owning storefront is unclear, create a separate `phone` entity with `region_hint` such as "unassigned visible phone/text" instead of attaching it to a nearby POI.

If a globally or nationally famous landmark/building is directly recognizable from the image, represent it as an `observed_entities` item with `entity_type: "landmark"`. This includes landmarks such as Tiananmen, the White House, the Great Hall of the People, the Eiffel Tower, Chiang Kai-shek Memorial Hall, or other widely known structures.

Only use `landmark` when recognition is grounded in visible evidence (distinctive architecture, silhouette, facade, monument layout, visible name fragments, symbols, or plaques). If it only resembles a famous landmark, describe the resemblance in `visual_cues` and `reasoning_notes` instead of naming a landmark entity.

## Cue Type Hierarchy

Use these exact `cue_type` values:

| Cue Type | Use For |
| --- | --- |
| `driving_side` | visible traffic direction, lane side, vehicle flow |
| `licence_plate` | plate format, color, character style, mounting |
| `road_marking` | lane lines, arrows, crosswalks, stop lines, painted road text |
| `road_sign` | traffic signs - observe shape, color, border, symbol style, text, and the count/layout of internal elements (e.g. stripe count on pedestrian signs), not just the sign type |
| `crossing` | pedestrian crossings, signals, intersection geometry |
| `bollard` | traffic bollards, barriers, cones, delineators |
| `guardrail` | road guardrails and roadside barriers |
| `utility_pole` | electric/telephone poles, overhead wires, pole layout |
| `lamp_post` | street lamp style, height, spacing |
| `kerb` | curb or sidewalk edge style, color, material |
| `bin` | waste bins or recycling containers |
| `taxi` | taxi color, roof signs, markings |
| `bus` | bus livery, route display, stops |
| `architecture` | building materials, roof forms, facade style, density |
| `vegetation` | trees, plants, gardens, climate-relevant flora |
| `terrain` | slopes, mountains, coastlines, rivers, lakes, flatness |
| `soil` | dirt, gravel, sand, paving, exposed ground |
| `climate` | weather, humidity, snow, dryness, lighting conditions |
| `language_script` | scripts and languages visible in text |
| `urban_form` | street density, development type, road width, neighborhood form |
| `landmark` | notable towers, monuments, unique structures |
| `bridge` | bridges, overpasses, elevated roads |
| `road` | road material, width, lane count, condition |
| `infrastructure` | drains, manholes, traffic lights, signals, cables |
| `coverage` | ground cover such as grass, concrete, water, forest |
| `follow_car` | visible capture vehicle or traffic-flow indicators |
| `rural_building` | farmhouses, barns, sheds, rural structures |
| `fire_hydrant` | hydrant style, color, placement |
| `rock_wall` | rock walls, cliffs, retaining walls |
| `other` | distinctive clue that does not fit another type |

## Traffic Sign Detail Observation

Traffic signs are high-value clues because their internal visual details vary by country/region. For every readable traffic sign, describe its concrete visual features, not just its type. Report each distinct sign as its own `visual_cues` entry with `cue_type: "road_sign"` and the concrete details in `text`.

- **Pedestrian crossing signs**: Accurately count the number of zebra stripes (like 3, 4, 5, 7, 8); note if stripes are replaced by dashed lines or a single horizontal line, or if they are missing; describe human figures – whether they have a belt (and its height), a hat, a square head, whether it’s a female or male silhouette, shoe details, a broken head, how detailed they are, and clothing; pay attention to the shape of signs (triangle, pentagon, rectangle), background color, and any colored borders.
- **Stop / yield signs**: color, shape, border, language and exact wording of the text.
- **Directional / chevron signs**: color scheme, font, arrow style, chevron count and density.
- **Town entry signs**: font, layout, color, language, name suffix style.
- **Kilometre markers / highway shields**: color, shape, number format, material, mounting.
- **Other warning / regulatory signs**: shape (triangle/diamond/circle), border color, background color, symbol style, any text.

Prefer precise counts and style descriptions over generic labels. "Pedestrian sign with 5 stripes, person wearing a belt sitting low" is far more valuable than "pedestrian crossing sign".

## Priority Guidelines

Prioritize geographically discriminative clues. Do not treat readable text as the only valuable evidence.

- Record named anchors precisely when they are visible: place names, business names, addresses, road names, postal codes, and directly recognizable landmarks.
- Independently record the strongest non-text regional clues: driving side, road-marking and sign design, utility infrastructure, vehicle and plate style, architecture and urban form, vegetation, terrain, climate, and coverage/capture characteristics.
- For architecture, vegetation, terrain, and cultural or commercial environment, state the concrete observable property and the geographic contrast it may suggest. For example, describe roof material, facade treatment, street density, crop type, mountain/coastal setting, or road furniture style; do not merely write "European architecture" or "tropical vegetation".
- A clear text anchor can justify later fine-grained investigation, but non-text clues must still be preserved because they establish country/region priors and can contradict a text-derived candidate.

## Uncertainty Handling

- Report uncertainty explicitly.
- If text is partly visible, preserve the visible characters and lower confidence.
- Copy visible text character-by-character; do not translate it unless needed in notes.
- Distinctive details are more valuable than obvious scene labels.
- Prefer several precise clues over many generic clues.
- Separate named/text anchors from regional visual clues in `reasoning_notes`. When the image has no reliable text anchor, explain which non-text clues are useful for proposing or excluding countries/regions and which are too generic to use.
- **Locatability estimate:** After extracting clues, assess the maximum granularity this image can realistically support. If the image contains named visual anchors (OCR place text, POI names, landmarks, addresses, road signs), set `max_expected_granularity` to `street`, `city`, or finer depending on specificity. If the image contains only generic scene clues (architecture, vegetation, traffic patterns) with no text or named anchors, set it to `country` or `continent`. The `score` reflects confidence in this ceiling (0.0-1.0). This estimate helps the reasoning agent allocate search effort appropriately.

## Visual Analysis Field

Use this structure for the `visual_analysis` field inside BrainDecision. Include every child field on the initial decision; use an empty array when nothing is observed for a list field.
```json
{
  "visual_analysis": {
  "scene_summary": "2-3 sentence factual description of the entire scene",
  "visual_cues": [
    {
      "cue_type": "exact cue type from the hierarchy above",
      "text": "specific observable clue with concrete details",
      "confidence": 0.0
    }
  ],
  "ocr_results": [
    {
      "text": "exact visible text copied character-by-character",
      "language": "detected language or null",
      "confidence": 0.0
    }
  ],
  "observed_entities": [
    {
      "id": "stable id such as ent_left_store",
      "entity_type": "poi|landmark|phone|address|sign|vehicle|other",
      "name": "visible POI/business/sign name, directly recognized landmark name, or null",
      "text_items": ["all text visibly tied to this same entity"],
      "phones": ["phone numbers only if visually tied to this same entity"],
      "region_hint": "where this entity appears in the image",
      "confidence": 0.0
    }
  ],
  "search_queries": [
    "2-4 concise web search queries combining the most distinctive clues"
  ],
  "reasoning_notes": [
    "what is directly observable, what is uncertain, and which clues are most useful"
  ],
  "locatability": {
    "max_expected_granularity": "coordinates|poi|street|city|region|country|continent",
    "score": 0.0,
    "rationale": "brief explanation of why this is the maximum expected precision given visible anchors and clues"
  }
  }
}
```

On later decisions, set `visual_updates` to null. If the recorded evidence is insufficient, request `visual_reanalysis` with one concrete question and target region or object; its result is merged into state automatically.
