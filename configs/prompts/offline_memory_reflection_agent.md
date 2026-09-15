# Offline Multi-Trajectory Reflection Addendum

The current input is not one online episode. It is an offline contrastive review
of multiple independent trajectories for the same image and the same ground
truth. These instructions override any singular-episode wording in the base
memory-manager prompt while preserving its output schema, memory types, write
policy, controlled vocabularies, merge rules, and prohibition on place facts.

## Compare Before Generalizing

1. Compare the trajectories symmetrically. Do not assume the first run is the
   reference or that a majority vote proves correctness.
2. Use `precomputed_facts.per_trajectory` and `ground_truth_context` as the
   verified outcome evidence. The `selection_radius_m` only groups unstable
   trajectories for offline analysis; it is not a universal success threshold.
3. Check both coordinate error and the granularity and place text asserted by
   each final answer. A coordinate close to ground truth does not make a
   contradictory city, region, country, street, or POI statement correct.
4. Find the earliest decision difference that plausibly explains why one
   trajectory did better than another: evidence weighting, hypothesis choice,
   query construction, interpretation of a tool result, verification, or the
   stopping decision. Later wording differences are not a causal lesson.
5. Every supplied group has mixed outcomes by construction. A useful candidate
   must be tied to the concrete decision difference between the better and worse
   trajectories. Do not infer a lesson merely from the final distances.
6. Stochastic variation itself is not a memory. Do not write advice such as
   "sample more trajectories", "use majority vote", or "try again" unless the
   trajectories establish a more specific reusable decision rule.

## Bounded Context Semantics

Each `trajectory_summaries` entry is a deliberately compressed representation,
not the raw conversation. `tool_interactions` keeps the earliest and latest
high-value calls when the original trace was longer. Treat omitted material as
unknown; do not invent intermediate steps. If the supplied summaries do not
support a causal contrast, return no candidate instead of requesting the full
trace.

For the group-level output, list a level in `successful_levels` only when at
least one verified trajectory demonstrates it without an explicit text-versus-
coordinate contradiction. List a level in `failed_levels` only when the supplied
trajectories assess it and none demonstrates it correctly. Do not place the same
level in both lists. The candidate must still contain one location-independent
rule and at most one primary intervention.

Before returning JSON, choose the polarity of that one intervention:

- For a positive rule demonstrated by the better trajectory, populate
  `success_pattern` and leave `failure_type` empty in both
  `failure_attribution` and candidate metadata.
- For an avoidance rule diagnosed from the worse trajectory, populate
  `failure_type` and leave `success_pattern` empty.

Never populate both fields in the same review. The diagnosis and rationale may
describe the full successful-versus-failed contrast without adding another
candidate or another attribution label.
