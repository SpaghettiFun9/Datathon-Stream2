# Required disclosure

| Item | Statement |
|---|---|
| External datasets | **None.** Only the three competition files (`train.csv`, `test(1).csv`, `sample_submission.csv`) were used. No attempt was made to identify, locate or download the original source dataset. |
| External code / notebooks / public solutions | **None** consulted or adapted. All code in this repository was written for this competition. |
| Pretrained models | **None.** All models are trained from scratch on `train.csv` by `scripts/run_final.py`. |
| AI tools / coding agents | **Yes.** Claude Code (Anthropic) was used as a coding assistant to write and run the pipeline under the team's direction: the problem framing, the validation design, the decision to exclude non-causal ("lead") features from the submission, and the choice of final model were team decisions and are documented in `METHODOLOGY.md`. All generated code was reviewed and is fully reproducible from this repository. |
| Manual modification / post-processing of predictions | **None**, other than what the code does automatically: predictions are clipped at 0 (concentrations cannot be negative) and library outputs are averaged with the weights in `results/final_model_info_A.json`. No hand edits to `submission.csv`. |
| Additional information beyond the competition files | **None.** In particular, no information from the hidden test targets, no external weather or pollution data, and no data from the test file's *later* rows (only rows at or before each observation hour feed the submitted predictions). |

Provenance chain that the reviewers can verify:

```
train.csv + test(1).csv
  -> src/data.py       (load, integrity checks)
  -> src/features.py   (deterministic feature generation, ~2 s)
  -> scripts/run_final.py  (train LightGBM / XGBoost / CatBoost on all train rows, predict test)
  -> submission.csv    (merged onto sample_submission.csv ids)
```
