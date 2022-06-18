 BASE OF PRJECT: 
# SampleFactory APPO baseline for Iglu


## Installation
For this baseline version uses branch ```for_baseline``` from Iglu gridworld repository. You can install this version by the following command:

```bash
pip install git+https://github.com/iglu-contest/gridworld.git@for_baseline
```

Just install all dependencies using:
```bash
pip install -r docker/requirements.txt
```

## Training APPO
Just run ```train.py``` with config_path:
```bash
python main.py --config_path iglu_baseline.yaml
```
## Enjoy baseline
Run ```enjoy.py``` :
```bash
python utils/enjoy.py
```

## Per-skill aggregation of the baselines performance metrics. 
For each task, we calculate F1 score between built and target structures. 
For each skill, we average the performance on all targets requiring that skill.

| F1 score        | flying |tall |diagonal | flat   | tricky | all  |
|-----------------| ----- | -----| -------|--------|-------|------|
| MHB agent (NLP) | 0.292 | 0.322 | 0.242  |  0.334 | 0.295 | 0.313 |
| MHB agent (full)| 0.233 |0.243  | 0.161  |0.290   |  0.251|  0.258|
| Random agent (full)| 0.039|0.036  | 0.044  |0.038   |  0.043|  0.039|

