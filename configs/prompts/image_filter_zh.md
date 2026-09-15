# 拼接街景图像第一轮过滤

你正在为后续的视觉地理定位和记忆学习流程筛选图像。输入是来自同一个街景位置的连续拼接全景图或多视角拼接图。检查完整全景或所有分图，并将它们视为同一位置的证据。同一标牌或实体出现在多个视角中时，需要去重。

你的任务仅仅是判断图像是否包含真实场景中可见、可读、可检索，并且能够支持后续地理定位的文字锚点。不要直接定位图像。不要预测国家、城市、地名、坐标或难度等级。不要利用外部知识补全、翻译或扩展图像中没有清晰显示的文字。

## 有效证据

- 有名称的 POI 或商户、机构、学校、医院、车站和地标。
- 道路、街道、高速公路或路线的名称和编号。
- 与某个地点存在明确视觉关联的地址或门牌号。
- 电话号码只有在其明显属于同一标牌、店面或其他包含位置相关文字的实体时才算有效。
- 同一图像中的多个不同命名实体很有价值，必须分别记录。

不要把 Google、百度或其他街景界面文字、水印、版权信息、导航控件、指南针标记、时间戳或其他数据集叠加内容视为场景文字。不要把单独出现的“餐厅”“酒店”或“药店”等通用类别词视为命名锚点。车辆品牌、商品广告和无关文字不能作为 POI 锚点，除非图像明确表明它们属于一个固定的真实地点。

## 证据与文字转录规则

- 只能使用输入图像中可见的证据。
- 按照图中实际内容准确转录文字，不要擅自纠正拼写。
- 部分可读文字中的不确定字符使用 `[?]` 表示。
- 不要编造缺失的单词、分店名称、地址或电话号码数字。
- 将明显属于同一店面、标牌或实体的文字归入同一个实体。
- 如果无法确定电话号码或地址属于哪个实体，将其单独记录，并把 `text_scene_binding` 设为 `ambiguous`。
- 同一个实体在多个视角中重复出现时只计数一次。只有至少两个不同的、有名称且与位置相关的实体可见时，`multi_poi` 才为 true。

## 标签定义

- 当真实场景中存在文字时，`has_scene_text` 为 true，即使文字无法辨认；界面叠加内容不计入。
- 当至少存在一个上述有效位置锚点时，`has_named_anchor` 为 true。有名称的道路或路线，以及足够完整的地址，也算作锚点；单独的通用类别词不算。
- `text_legibility`：重要锚点文字能够可靠辨认时为 `clear`；仍有可用片段时为 `partial`；存在文字但无法用于检索时为 `unreadable`；真实场景中没有文字时为 `none`。
- `searchability`：具有独特完整名称或足够完整地址时为 `high`；具有可用的部分名称、道路与地址组合，或者与实体明确关联的电话证据时为 `medium`；文字线索较弱或基本属于通用信息时为 `low`；无法根据可见证据构造检索词时为 `none`。
- `specificity`：可见文字看起来能够确定某个特定地点时为 `unique`；可能对应多个分店或地点时为 `ambiguous`；仅为宽泛或常见名称、类别时为 `generic`；没有命名锚点时为 `none`。
- `image_quality`：相关区域清晰时为 `good`；存在模糊、距离过远、遮挡、眩光或拼接伪影，但仍保留有效证据时为 `usable`；这些问题导致无法可靠筛选时为 `poor`。
- 只有同时满足以下条件时，`filter_pass` 才能为 true：存在命名锚点；`text_legibility` 为 `clear` 或 `partial`；`searchability` 为 `high` 或 `medium`；`image_quality` 为 `good` 或 `usable`。不要仅仅因为锚点可能属于连锁品牌、存在重名、使用非拉丁文字或图中存在多个 POI 就拒绝图像，这些都是有价值的抽样分层。

按照以下顺序选择第一个匹配的 `recommended_stratum`：

1. 可见两个或更多不同命名实体时，选择 `multi_poi`。
2. 主要名称较为通用或可能存在多个分店时，选择 `branch_or_generic_name`。
3. 主要有效锚点是道路、地址、路线或与实体关联的电话时，选择 `road_address_or_phone`。
4. 主要锚点只能部分辨认，或者主要使用非拉丁文字时，选择 `partial_or_non_latin_text`。
5. 其他具有明确命名锚点并通过筛选的图像，选择 `specific_named_anchor`。
6. `filter_pass` 为 false 时，选择 `reject`。

只返回一个有效的 JSON 对象，不要输出 Markdown 或解释性文字。没有内容时使用空数组，只有格式明确允许时才能使用 `null`。`anchor_types` 只包含实际观察到的类型，`quality_issues` 只包含实际存在的问题；下面的格式列出的是允许值。`visible_text` 只记录与位置有关的文字。置信度必须是 0.0 到 1.0 之间的数字。

确保字段内部一致：`entity_count` 必须等于 `observed_entities` 的长度；`multi_poi` 必须遵守上述不同实体判定规则；当且仅当 `filter_pass` 为 false 时，`recommended_stratum` 才能为 `reject`。

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
  "visible_text": ["图像中准确可见的文字"],
  "searchable_text": ["适合后续检索的准确可见文字"],
  "text_legibility": "clear|partial|unreadable|none",
  "searchability": "high|medium|low|none",
  "specificity": "unique|ambiguous|generic|none",
  "language_or_script": ["语言或文字系统名称，无法判断时为 unknown"],
  "observed_entities": [
    {
      "id": "ent_1",
      "entity_type": "poi_business|institution|landmark|road_street|route_sign|address|phone|other_named_sign",
      "name": "准确可见的实体名称或 null",
      "text_items": ["属于该实体的准确可见文字"],
      "phones": ["准确可见的电话号码"],
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
  "filter_reason": "一句简短且基于可见证据的说明",
  "screening_confidence": 0.0
}
