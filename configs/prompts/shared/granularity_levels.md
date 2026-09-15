# Shared Granularity Levels

Use the most precise granularity that is actually supported by evidence.

From most to least precise:

1. `coordinates`: latitude/longitude with a limited uncertainty radius; use when evidence supports the image location through the camera position, a visible POI, a supported full house-number address, or a specific entrance/nearby anchor point
2. `poi`: named landmark, venue, station, store, or facility
3. `street`: street, road, intersection, or neighborhood-scale corridor; lat/lon may be a reference point for the street, not the exact camera position
4. `city`: specific city or metropolitan area
5. `region`: state, province, prefecture, district, island, or broad administrative region
6. `country`: country-level inference only
7. `continent`: continent or very broad world region
8. `unknown`: no meaningful geographic inference

Hypotheses and intermediate evidence may omit coordinates when they are not justified. A Brain `final_answer` follows the final-answer contract: provide the best numeric coordinate estimate while keeping `granularity`, confidence, and uncertainty calibrated to the evidence.

Street-level geocoding often returns a representative point, center point, or viewport for the street. This supports `street` granularity, not `coordinates`, unless additional evidence identifies a specific visible object, POI, address, entrance, or near-camera anchor along the street. A supported full address with a house number, building, entrance, or specific POI can support `coordinates` even when street-view coverage is unavailable.
