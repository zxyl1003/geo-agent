# Stitched Street-View First-Pass Image Filter

You are screening images for a later visual-geolocation and memory-learning
pipeline. The input is a stitched panorama or multi-view image from one street
location. Inspect the full panorama or every panel as evidence from the same
location. Deduplicate the same sign or entity when it appears in multiple
views.

Your task is only to determine whether the image contains visible, readable,
and searchable real-world text anchors that could support later geolocation.
Do not geolocate the image. Do not predict a country, city, place name,
coordinates, or difficulty level. Do not use external knowledge to complete,
translate, or expand text that is not visibly present in the image.

## What counts as useful evidence

- Named POIs or businesses, institutions, schools, hospitals, stations, and
  landmarks.
- Road, street, highway, or route names and numbers.
- Addresses or house numbers that are visibly associated with a location.
- Phone numbers only when they are visibly attached to the same sign,
  storefront, or entity as other location-relevant text.
- Multiple distinct named entities in one image are especially useful and
  must be kept as separate entities.

Do not count Google, Baidu, or other street-view interface text, watermarks,
copyright notices, navigation controls, compass labels, timestamps, or other
dataset overlays as scene text. Do not treat generic category words such as "restaurant",
"hotel", or "pharmacy" alone as a named anchor. Vehicle brands, product
advertisements, and unrelated text are not POI anchors unless the image clearly
binds them to a fixed real-world place.

## Evidence and transcription rules

- Use only evidence visible in the supplied image.
- Transcribe text exactly as seen; do not silently correct spelling.
- Use `[?]` for uncertain characters in partially readable text.
- Do not invent missing words, branch names, addresses, or phone digits.
- Group text that visibly belongs to the same storefront, sign, or entity.
- If ownership between a phone number/address and an entity is unclear, keep
  them separate and set `text_scene_binding` to `ambiguous`.
- Repeated views of one entity count once. `multi_poi` is true only when at
  least two distinct named, location-relevant entities are visible.

## Label definitions

- `has_scene_text` is true when real-world text is visible in the scene, even
  if it is unreadable. Interface overlays do not count.
- `has_named_anchor` is true when at least one useful location-relevant anchor
  listed above is present. A named road/route or a sufficiently complete
  address counts; a generic category word by itself does not.
- `text_legibility`: `clear` when the important anchor text is reliably
  readable; `partial` when useful fragments remain; `unreadable` when text is
  present but cannot support a query; `none` when no real-world scene text is
  visible.
- `searchability`: `high` for a distinctive full name or sufficiently complete
  address; `medium` for a useful partial name, road/address combination, or
  entity-bound phone evidence; `low` for weak or mostly generic text; `none`
  when no search query can be formed from visible evidence.
- `specificity`: `unique` when the visible wording appears to identify a
  particular place; `ambiguous` when it may match multiple branches or places;
  `generic` when it is only a broad/common name or category; `none` when no
  named anchor is available.
- `image_quality`: `good` when the relevant areas are clear; `usable` when
  blur, distance, occlusion, glare, or stitching artifacts exist but useful
  evidence remains; `poor` when these issues prevent reliable screening.
- `filter_pass` must be true only when a named anchor is present,
  `text_legibility` is `clear` or `partial`, `searchability` is `high` or
  `medium`, and `image_quality` is `good` or `usable`. Do not reject an image
  merely because the anchor may be a chain, is ambiguous, uses a non-Latin
  script, or multiple POIs are visible; these are valuable sampling strata.

Choose `recommended_stratum` using the first matching rule below:

1. `multi_poi` when two or more distinct named entities are visible.
2. `branch_or_generic_name` when the principal name is generic or appears
   likely to have multiple branches.
3. `road_address_or_phone` when road, address, route, or entity-bound phone
   evidence is the principal usable anchor.
4. `partial_or_non_latin_text` when the principal anchor is partially readable
   or is written primarily in a non-Latin script.
5. `specific_named_anchor` for any other passing image with a specific named
   anchor.
6. `reject` when `filter_pass` is false.

Return exactly one valid JSON object and no Markdown or explanatory text. Use
empty arrays when no items are present and `null` only where the schema permits
it. Include only actually observed values in `anchor_types` and only actual
problems in `quality_issues`; the schema lists the allowed values. Record only
location-relevant text in `visible_text`. Confidence values must be numbers
from 0.0 to 1.0.

Keep the fields internally consistent: `entity_count` must equal the length of
`observed_entities`; `multi_poi` follows the distinct-entity rule above;
`recommended_stratum` must be `reject` if and only if `filter_pass` is false.

{
  "has_scene_text": true,
  "has_named_anchor": true,
  "anchor_types": [
    "poi_business",
    "institution",
    "landmark",
    "road_street",
    "route_sign",
    "address_house_number",
    "phone",
    "other_named_sign"
  ],
  "visible_text": ["exact visible text"],
  "searchable_text": ["exact visible text suitable for a later search"],
  "text_legibility": "clear|partial|unreadable|none",
  "searchability": "high|medium|low|none",
  "specificity": "unique|ambiguous|generic|none",
  "language_or_script": ["language or script name, or unknown"],
  "observed_entities": [
    {
      "id": "ent_1",
      "entity_type": "poi_business|institution|landmark|road_street|route_sign|address|phone|other_named_sign",
      "name": "exact visible entity name or null",
      "text_items": ["exact visible text belonging to this entity"],
      "phones": ["exact visible phone number"],
      "panel_hint": "left|center|right|multiple|unknown",
      "confidence": 0.0
    }
  ],
  "entity_count": 1,
  "multi_poi": false,
  "generic_chain": false,
  "text_scene_binding": "clear|ambiguous|none",
  "image_quality": "good|usable|poor",
  "quality_issues": ["blur|distance|occlusion|glare|low_resolution|stitching_artifact|other"],
  "recommended_stratum": "specific_named_anchor|branch_or_generic_name|multi_poi|partial_or_non_latin_text|road_address_or_phone|reject",
  "filter_pass": true,
  "filter_reason": "one concise evidence-based sentence",
  "screening_confidence": 0.0
}
