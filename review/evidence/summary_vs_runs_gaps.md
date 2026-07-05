# Summary vs. run-directory gap check

Total run directories under `results/predictions/`: 56

Rows in `results/summary/final_table.csv`: 28

Rows in `results/summary_ner/final_table_ner.csv`: 5


## Runs present in results/predictions/ but MISSING from a summary CSV

| run_id | dataset | missing_from | note |
|---|---|---|---|
| finbert__FPBall__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| finbert_ner__FiQA__seed42 | FiQA | BOTH | exploratory NER/tone track (entities.jsonl or tone-classifier) -- not part of the FiNER-ORD NER benchmark or sentiment final_table |
| finbert_ner__FPB__seed42 | FPB | BOTH | exploratory NER/tone track (entities.jsonl or tone-classifier) -- not part of the FiNER-ORD NER benchmark or sentiment final_table |
| finbert_tone__FiQA__seed42 | FiQA | BOTH | exploratory NER/tone track (entities.jsonl or tone-classifier) -- not part of the FiNER-ORD NER benchmark or sentiment final_table |
| finbert_tone__FPB__seed42 | FPB | BOTH | exploratory NER/tone track (entities.jsonl or tone-classifier) -- not part of the FiNER-ORD NER benchmark or sentiment final_table |
| mistral7b__FPBall__A__0shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| mistral7b__FPBall__A__3shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| mistral7b__FPBall__B__0shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| mistral7b__FPBall__B__3shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| ner__mistral7b__FIN__paper__0shot__seed42 | FIN | BOTH | FIN NER benchmark run (98-doc CoNLL-style set) -- not represented in either final_table CSV |
| ner__plutus8b__FIN__paper__0shot__seed42 | FIN | BOTH | FIN NER benchmark run (98-doc CoNLL-style set) -- not represented in either final_table CSV |
| ner__qwen25_7b__FIN__paper__0shot__seed42 | FIN | BOTH | FIN NER benchmark run (98-doc CoNLL-style set) -- not represented in either final_table CSV |
| plutus8b__FPBall__A__0shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| plutus8b__FPBall__A__3shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| plutus8b__FPBall__B__0shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| plutus8b__FPBall__B__3shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| qwen25_7b__FPBall__A__0shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| qwen25_7b__FPBall__A__3shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| qwen25_7b__FPBall__B__0shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| qwen25_7b__FPBall__B__3shot__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |
| qwen3_4b__FiNER-ORD__A__0shot__seed42 | FiNER-ORD | final_table_ner.csv | key ('qwen3_4b', 'FiNER-ORD', 'A', 0) not found in final_table_ner.csv |
| qwen3_8b__FiNER-ORD__A__0shot__seed42 | FiNER-ORD | final_table_ner.csv | key ('qwen3_8b', 'FiNER-ORD', 'A', 0) not found in final_table_ner.csv |
| vader__FPBall__seed42 | FPBall | final_table.csv | FPBall (full-agreement-relaxed FPB variant) run -- not a column dataset in final_table.csv (which only has FPB/FiQA) |

## Rows in summary CSVs with NO matching prediction directory


### final_table.csv

_None -- every row has a matching run directory._


### final_table_ner.csv

_None -- every row has a matching run directory._


## Additional discrepancy noted during census

- `final_table_ner.csv` reports `n_samples=200` for `mistral7b`, `plutus8b`, and `qwen25_7b` on `FiNER-ORD`, but the corresponding `results/predictions/{model}__FiNER-ORD__A__0shot__seed42/` directories contain 300 lines in `predictions.jsonl` and `meta.json` / `progress.json` both say `n_total=300`. This suggests the NER summary aggregation step may have restricted to a 200-sample subset (or dropped rows) for the LLM rows while gliner-large/gliner-small kept 300. Flagging for the next review stage -- not resolved here.
