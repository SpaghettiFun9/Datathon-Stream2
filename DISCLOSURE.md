# Required disclosure

| Item | Statement |
|---|---|
| External datasets | **None.** Only the three competition files (`train.csv`, `test(1).csv`, `sample_submission.csv`) were used. No attempt was made to identify, locate or download the original source dataset. |
| External code / notebooks / public solutions | **None** consulted or adapted. All code in this repository was written for this competition. |
| Pretrained models | **None.** All models are trained from scratch on `train.csv` by `scripts/run_final.py`. |
| AI tools / coding agents | **Yes.** Claude Code (Anthropic) was used as a coding assistant to write and run the pipeline under the team's direction: the problem framing, the validation design, the decision to exclude non-causal ("lead") features from the submission, and the choice of final model were team decisions and are documented in `METHODOLOGY.md`. All generated code was reviewed and is fully reproducible from this repository. |
| Manual modification / post-processing of predictions | **None**, other than what the code does automatically: predictions are clipped at 0 (concentrations cannot be negative) and library outputs are averaged with the weights in `results/final_model_info_LS.json`. No hand edits to `submission.csv`. |
| Additional information beyond the competition files | **None** from outside the files: no hidden test targets, no external weather or pollution data. **Within the files:** the submitted model (variant L) uses, for the row observed at hour *t*, the predictor readings of the rows at *t+1 … t+6* of the same test file (never any target). Our causal alternative, which uses only rows at or before *t*, is also provided (`results/experiments/submission_causal_LB23.43549.csv`, public LB 23.43549) and is documented with identical rigour; see README "Two model families" and METHODOLOGY.md §7. |

Provenance chain that the reviewers can verify:

```
train.csv + test(1).csv
  -> src/data.py       (load, integrity checks)
  -> src/features.py   (deterministic feature generation, ~2 s)
  -> src/stacking.py   (stage-1 nowcast, out-of-fold on train / full-fit on test)
  -> scripts/run_final.py  (train LightGBM / XGBoost / CatBoost on all train rows, predict test)
  -> submission.csv    (merged onto sample_submission.csv ids)
```
