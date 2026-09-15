You are performing a single-image geolocation benchmark. Analyze the entire
image carefully and infer where it was captured. Use only visible evidence in
the image and your pretrained knowledge; do not use external tools.

Consider all useful geographic cues, including readable text, language, road
signs and markings, traffic direction, architecture, infrastructure, vehicles,
terrain, vegetation, climate, and distinctive landmarks. Compare plausible
locations internally, then commit to one best estimate. Do not default to a
capital or famous city unless the visual evidence supports it.

Always provide one best WGS84 coordinate estimate, even when the evidence is
weak. The country, city, and coordinates must describe the same location. For
rural locations, use the nearest reasonable city or municipality. Do not
return alternatives, ranges, `null`, or an explanation.

Return exactly one JSON object and no Markdown, using this schema:

{
  "country": "best-estimate country name in English",
  "city": "best-estimate city or municipality name in English",
  "latitude": 0.0,
  "longitude": 0.0
}

`latitude` and `longitude` must be JSON numbers, not strings. Latitude must be
between -90 and 90; longitude must be between -180 and 180.
