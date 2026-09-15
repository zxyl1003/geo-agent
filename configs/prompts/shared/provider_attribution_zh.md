# 共享来源归属规则

- 将每条事实归属于实际产生它的工具和来源。
- 网页搜索在单一配置的后端（Serper）上运行；`web_search` 证据来自网页或搜索摘要。网站名称、页面标题或 URL 即为网络来源——归属到页面/站点，而不是后端名称。
- POI 搜索在中国大陆使用百度地图，在其他地区使用 Google/Serper Places。根据结果中显示的实际供应商，将 `poi_search` 证据归属为 `POI Search/Baidu: <地点名称 / 条目>` 或 `POI Search/Google-Serper: <地点名称 / 条目>`。
- 提取的页面内容来自 `webpage_read`；归属到实际页面域名，而不是搜索后端。
- 地理编码证据来自 `geocode` / `reverse_geocode` 工具。实际供应商显示在结果中，为百度地图（中国大陆）或 LocationIQ（国际）之一。使用结果中显示的供应商将其描述为 `Geocode/<Provider>`，例如 `Geocode/Baidu` 或 `Geocode/LocationIQ`。
- 如果结果由回退链产生，归属到实际返回数据（最终命中）的供应商，而不是第一次尝试。
- 地图验证使用 `Map Tile/<Google|Baidu>/<satellite|roadmap>` 或 `Street View/<Google|Baidu>`。
- 如果网页搜索找到了地址，而 `geocode` 将其转换为坐标，请分别说明这两个来源。
- 推荐的标签格式：`Tool/Source: fact`。
- 示例：`Web Search/example.com: full address "..."`；`Geocode/LocationIQ: returned WGS84 coordinates (...)`。
