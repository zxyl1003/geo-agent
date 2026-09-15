# Shared Provider Attribution Rules

- Attribute each fact to the tool and source that actually produced it.
- Web search runs on a single configured backend (Serper); `web_search` evidence comes from web pages or search snippets. A website name, page title, or URL is a web source — attribute the page/site, not the backend name.
- POI search uses Baidu Maps in mainland China and Google/Serper Places elsewhere. Attribute `poi_search` evidence as `POI Search/Baidu: <place name / listing>` or `POI Search/Google-Serper: <place name / listing>` according to the actual provider shown in the result.
- Extracted page content comes from `webpage_read`; attribute the actual page domain, not the search backend.
- Geocoding evidence comes from the `geocode` / `reverse_geocode` tools. The actual provider is shown in the result and is either Baidu Maps (mainland China) or LocationIQ (international). Describe it as `Geocode/<Provider>` using the provider shown in the result, e.g. `Geocode/Baidu` or `Geocode/LocationIQ`.
- If a result was produced by a fallback chain, attribute the provider that actually returned the data (the final provider), not the first attempt.
- For map verification, use `Map Tile/<Google|Baidu>/<satellite|roadmap>` or `Street View/<Google|Baidu>`.
- If web search finds an address and `geocode` converts it to coordinates, state both sources separately.
- Preferred label format: `Tool/Source: fact`.
- Example: `Web Search/example.com: full address "..."`; `Geocode/LocationIQ: returned WGS84 coordinates (...)`.
