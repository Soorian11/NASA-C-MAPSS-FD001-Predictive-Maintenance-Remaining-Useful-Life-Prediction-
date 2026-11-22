# NASA C-MAPSS FD001 Remaining Useful Life Project

## 1. Project overview

This project estimates the remaining useful life (RUL) of simulated turbofan engines. RUL is the number of operating cycles remaining before the engine reaches the end of its recorded run-to-failure trajectory. The project uses FD001, one subset of NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) data.



## 2. What the project does

The end-to-end task is a supervised regression problem:

1. Read the training trajectories, test trajectories, and official test RUL labels.
2. Validate the raw data schema and engine/cycle identifiers.
3. Calculate a capped training target, `RUL = final_cycle - current_cycle`, with a maximum of 125 cycles.
4. Create time-aware sensor features within each engine trajectory.
5. Fit candidate models and select a model using validation data.
6. Refit the selected model on the training trajectories.
7. Predict RUL at the last recorded cycle of each test engine.
8. Report errors and save predictions, plots, and a reusable model.

The two entry points have different levels of validation rigor. The Colab notebook uses multiple simulated censoring points per held-out engine and grouped cross-validation for tuning. The local `src/train.py` script currently scores only the final row of each held-out run-to-failure training engine. Those rows all have RUL 0, so its validation comparison does not represent the censored test problem and should not be used as a reliable model-selection estimate. This limitation is described here so that the local script's output is interpreted correctly.

## 3. Dataset

FD001 has 100 run-to-failure training engines and 100 censored test engines. It represents one operating condition and one simulated fault mode. Training paths end at failure. Test paths stop before failure; `RUL_FD001.txt` supplies the remaining cycles for each test engine's final observation.

Each trajectory row has 26 whitespace-separated numeric values:

| Column | Meaning |
|---|---|
| `unit_id` | Engine identifier |
| `cycle` | Operating cycle number within that engine |
| `setting_1`–`setting_3` | Operating settings |
| `sensor_1`–`sensor_21` | Sensor measurements |

The project assigns names to these fields when loading the text files. Test RUL labels are in ascending engine-ID order; the workflows sort final test predictions by engine ID before combining predictions with those labels.

The data files are downloaded from a public Hugging Face mirror of C-MAPSS. The underlying dataset is associated with A. Saxena, K. Goebel, D. Simon, and N. Eklund (2008), “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,” PHM08.

### Observed characteristics of this FD001 copy

The supplied files contain 20,631 training rows and 13,096 test rows. Training trajectory length ranges from 128 to 362 cycles, with a median of 199. The official test RUL values range from 7 to 145 cycles, with a median of 86. Sensors 1, 5, 10, 16, 18, and 19 are effectively constant in this copy; operating setting 3 is constant. The notebook calculates and displays these observations from the files rather than assuming they hold for other C-MAPSS subsets.


### Local training methodology

The local script caps row-level training targets at 125, creates a one-time 80/20 engine-grouped split, compares models by MAE, fits the model with the best split MAE on all training rows, then evaluates on the official test set. However, the local validation endpoint is each held-out training engine's last row. Because those trajectories run to failure, the target at every such endpoint is zero. This is a known methodological limitation of this compact Python path. The local validation CSV is generated for transparency, but do not interpret its rankings as comparable evidence for performance on censored test endpoints.

For model selection that better reflects the official test format, use the companion Colab notebook described in Section 7. It selects checkpoints before failure and tunes using engine-grouped folds.

## 7. Google Colab notebook workflow


1. **Setup and loading:** set a fixed random seed, download the files, validate schema, missing values, positive cycle numbers, duplicate engine/cycle pairs, and label count.
2. **Exploratory analysis:** inspect dataset dimensions, trajectory lengths, capped RUL distribution, sensor variability, operating-setting variability, and selected sensor trajectories.
3. **Feature engineering:** add current measurements, cycle count, within-engine first differences, trailing means over 5/10/20 cycles, and trailing standard deviations over 5/10 cycles.
4. **Outer validation split:** reserve 20% of engine IDs for validation. No engine appears in both the outer fitting set and outer validation set.
5. **Simulated censoring:** create checkpoints at 0, 25, 50, 75, 100, and 125 cycles before failure for each held-out engine. This avoids evaluating only the zero-RUL run-to-failure endpoint.
6. **Hyperparameter tuning:** sample eight HistGradientBoosting parameter combinations and compare each with four-fold `GroupKFold` using training engines only. Each fold is scored at the same simulated censoring checkpoints.
7. **Model comparison:** compare a median-RUL baseline, scaled Ridge regression, and the tuned HistGradientBoosting model on the untouched outer validation engines. Report MAE, RMSE, R², and the NASA asymmetric score.
8. **Official test evaluation:** refit the selected tuned model using all training engines and evaluate once at each official test engine's final observed cycle.
9. **Interpretation and artifacts:** plot actual/predicted values and errors, calculate permutation importance on the outer validation set, save the final model and reports, and provide an optional new-history prediction section.


### Validation design details

The training set contains full run-to-failure histories, so labels at intermediate cycles can be calculated from the known failure cycle. The notebook uses those labels to emulate a test engine observed early. For a requested checkpoint `r`, it selects the row at `final_cycle - r` and uses the capped RUL label at that row. Features at that row are trailing-only, so later sensor measurements are not included in the prediction.

The grouped folds are made over engine IDs, not rows. This prevents neighboring cycles from the same engine from being split between fold fitting and scoring. The outer held-out engines remain separate from hyperparameter selection and provide an additional comparison before the final official test evaluation.

## 8. Target and feature definitions

For a training engine with final recorded cycle `T`, and current cycle `t`:

```text
raw_RUL = T - t
training_RUL = min(raw_RUL, 125)
```

The official test labels are not capped in the metric calculation. Model predictions are clipped to `[0, 125]` because the model is trained on the capped target.

Predictors include:

- the current cycle and operating settings;
- current readings for all 21 sensors;
- `cycles_from_start = cycle - 1`;
- sensor first differences from the previous cycle (first cycle uses 0);
- trailing sensor means over 5, 10, and 20 cycles;
- trailing sensor standard deviations over 5 and 10 cycles (not enough history uses 0).

All features are created independently within each engine and sorted by cycle. The lifetime-normalized cycle feature is intentionally not used, since dividing by an engine's eventual final cycle would expose future lifetime information to earlier rows.

## 9. Metrics

The notebook reports:

- **MAE:** mean absolute prediction error in cycles; lower is better.
- **RMSE:** square root of mean squared error in cycles; larger errors count more heavily.
- **R²:** variance explained relative to predicting the mean; higher is better, and it can be negative.
- **NASA asymmetric score:** underprediction and overprediction receive different exponential penalties; lower is better. The notebook reports the mean score per checkpoint for interpretability.

Metrics describe performance on these simulated FD001 trajectories only. Scores depend on the split, checkpoints, model settings, and capping choice. A score alone does not establish operational safety or real-world value.


