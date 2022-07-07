# SampleFactory APPO baseline for Iglu

## Idea
Training an agent to build any language-defined structure is a challenging task. To overcome this, we have developed
a multitask hierarchical builder (MHB) with three modules: task generator (NLP part), subtask generator (heuristic
part), and subtask solving module (RL part). We define the subtask as an episode of adding or removing a single cube. It allows us to train an agent with a dense reward signal in episodes with a short horizon.

**Task generator module** generate full target(3D voxel) figure using dialogue with person. For training we use randomly generated compact structures as tasks.

**Subtask generator** receives a 3D voxel as input and outputs a sequence of subgoals (remove or install one cube) in a certain sequence (left to right, bottom to top);

**Subtask solving module** APPO agent who is learning the task of adding or removing one cube.

Original figure (3D voxel)             |  Baseline building process
:-------------------------:|:-------------------------:
![Baseline job example](https://github.com/ZoyaV/multitask_baseline/raw/master/original.jpg) |  ![Baseline job example](https://github.com/ZoyaV/multitask_baseline/raw/master/example.gif)

## Code structure

Now, function `target_to_subtasks` in wrappers/target_generator.py implements the main algorithm for splitting the goal into subtasks
Also, in  `wrappers/multitask` you can find `TargetGenerator` and `SubtaskGenerator` classes.
First class make full-figure target using `RandomFigure` generator or `DatasetFigure` generator.
Second class make subtasks for environment.

## Installation
For this baseline version uses branch ```segments``` from Iglu gridworld repository. You can install this version by the following command:

```bash
pip install git+https://github.com/iglu-contest/gridworld.git@segments
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
Instead of evaluating a metric for each structure in the dataset, we evaluate the agent's skills required to build each structure.
There are 5 skills in total:
  * `flat` - flat structure with all blocks on the ground
  * `flying` - there are blocks that cannot be placed without removing some other blocks (i.e. )
  * `diagonal` - some blocks are adjacent (in vertical axis) diagonally
  * `tricky` - some blocks are hidden or there should be a specific order in which they should be placed
  * `tall` - a structure cannot be built without the agent being high enough (the placement radius is 3 blocks)
  
For each task, we calculate F1 score between built and target structures. 
For each skill, we average the performance on all targets requiring that skill.

| F1 score        | flying |tall |diagonal | flat   | tricky | all  |
|-----------------| ----- | -----| -------|--------|-------|------|
| MHB agent (NLP) | 0.292 | 0.322 | 0.242  |  0.334 | 0.295 | 0.313 |
| MHB agent (full)| 0.233 |0.243  | 0.161  |0.290   |  0.251|  0.258|
| Random agent (full)| 0.039|0.036  | 0.044  |0.038   |  0.043|  0.039|

## Results
![Baseline job example](https://github.com/ZoyaV/multitask_baseline/raw/master/bexampl.gif)
