"""NER runners — one entry per backend.

All runners implement:
    .predict_one(sample) -> Prediction dict
or
    .predict_many(samples) -> list[Prediction]

A Prediction dict has:
    id, pred_tags (list[str] | None), pred_entities (list[dict]),
    raw_output (str), parse_ok (bool), latency_ms (float),
    input_tokens (int), output_tokens (int), cost_usd (float)
"""
