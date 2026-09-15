# Shared Coordinate Contract

- All public `lat` and `lon` fields in compact state, tool results, hypotheses, final answers, and CSV exports are WGS84.
- Use public `lat` and `lon` for hypotheses, verification, and final answers.
- Provider-native coordinates may appear only as `raw_lat`, `raw_lon`, and `raw_coordinate_system`.
- Never use raw provider-native coordinates as final coordinates.
- Google coordinates are already WGS84.
- Baidu provider-native coordinates are usually BD-09; Baidu geocoding/POI tools normalize those results to public WGS84 fields.
