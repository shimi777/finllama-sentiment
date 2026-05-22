"""NER comparison module — separate from sentiment pipeline.

Mirrors the sentiment pipeline architecture but for token-level NER on
FiNER-ORD (PER / LOC / ORG) using modern LLMs (local GLiNER + cheap APIs).

Public surface:
- data_loader.load_finer_ord    -> list[NerSample]
- prompts.build_prompt          -> str
- parser.parse_json_to_bio      -> list[str] | None
- evaluation.compute_ner_metrics-> dict (seqeval-based)
- cost.CostTracker              -> per-run + cumulative USD guard
"""
