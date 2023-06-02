# HardSATGEN: Understanding the Difficulty of Hard SAT Formula Generation and A Strong Structure-Hardness-Aware Baseline

Official implementation of SIGKDD 2023 paper "HardSATGEN: Understanding the Difficulty of Hard SAT Formula Generation and A Strong Structure-Hardness-Aware Baseline".

## Installation

- Install PyTorch (tested on 1.0.0), please refer to the offical website for further details

```bash
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

- Install PyTorch Geometric (tested on 1.1.2), please refer to the offical website for further details

```bash
pip install --verbose --no-cache-dir torch-scatter
pip install --verbose --no-cache-dir torch-sparse
pip install --verbose --no-cache-dir torch-cluster
pip install --verbose --no-cache-dir torch-spline-conv (optional)
pip install torch-geometric
```

- Install networkx (tested on 2.3), make sure you are not using networkx 1.x version!

```bash
pip install networkx
```

- Install tensorboardx

```bash
pip install tensorboardX
```

- Build post-process toolbags
```bash
cd postprocess/cadical && ./configure && make
cd drat-trim && make && cd ../..
```


## Dataset preparation

example dataset: lec-0707
Needed files：
- cnf：put in `/dataset/lec-0707/`, named `aaa.cnf`, `bbb.cnf`, ...
- core：put in `/dataset/lec-0707_core/`, named `aaa_core`, `bbb_core`, ...
- lcg_stats.csv：exists in `/dataset/`, add each cnf informationswith format as `lec_0707/aaa, num_var, num_clause`

How to build dataset when only cnfs in hand:
1. Prepare core:make sure to build cadical & drat-trim first
```bash
bash solve_core.sh dataset/class_name
```
Core files will generated under dataset/class_name, move them into dataset/class_name_core.
Also, the cadical solving log are stored in dataset/class_name.log, check solving time if needed.

2. Prepare csv:
```bash
class_name/cnf_name,num_var,num_clause
```
The information must be correspounding to original cnf file.
If a cnf file can't be found in csv, then it will not be use to train/test.
** The cnf-lcg converting process may cause "Stats not match", model will automatically skip these cnfs. Since the reason are still not clarified, if encountered this in train/test, we recommend you replace the bad files with some files that didn't cause this problem.



## Example Run

You can try out the following 4 steps one by one. Instructions extracted from [https://github.com/JiaxuanYou/G2SAT](https://github.com/JiaxuanYou/G2SAT).

1. Train
```bash
# Do not remain unsat core
python main_train.py --epoch_num 201 --data_name train_set --model GCN # SAGE; GCN; EGNN; ECC
# remain unsat core && introduce scrambling operation
python main_train.py --epoch_num 201 --data_name train_set --core_flag --model GCN # SAGE; GCN; EGNN; ECC
```
After this step, trained G2SAT models will be saved in `model/` directory.

2. Use to generate Formulas
```bash
python main_test.py --epoch_load 200 --data_name train_set --model GCN # SAGE; GCN; EGNN; ECC
# remain unsat core && introduce scrambling operation
python main_test.py --epoch_num 200 --data_name train_set --core_flag --model GCN # SAGE; GCN; EGNN; ECC
```
After this step, generated graphs will be saved to `graphs/` directory. 1 graph is generated out of 1 template.

Graphs will be saved in a single `.dat` file containing all the generated graphs.

(It may take fairly long time: Runing is fast, but updating networkx takes the majority of time in current implementation.)

We can then generate CNF formulas from the generated graphs
```bash
python eval/conversion_lcg.py --src graphs/train_set_lcg_GCN_coreTrue_alpha.dat --store-dir formulas/train_set --action=lcg2sat
```

3. Post-processing for Formulas
```bash
cd postprocess && bash auto_rmcore.sh train_set 200 2255
# ${1}: class name in ./formulas/train_set
# ${2}: maximum re-post times
# ${3}: expected sovling time threshold

# or multi-task:
cd postprocess && bash auto_rmcore_multi.sh train_set 200 2255
```
The post-processed formulas will be stored in ./formulas/train_set_post, logs in ./postprocess/cadical/build


## Eval and solver tuning

1. Evaluate graph properties of formulas
```bash
# build eval tools
g++ -o eval/scalefree eval/scalefree.cpp

# evaluate
python eval/evaluate_graphs_lcg.py -s eval/scalefree -d formulas/train_set/ -o train_set.csv
# python eval/evaluate_graphs_fg.py -s eval/scalefree -d formulas/train_set/ -o train_set.csv
```

2. Solver tuning
Here we use glucose to test the tuning ability. (following the experiments settings of G2SAT)
Build glucose first. If the exist directory is broken, download new one from https://github.com/wadoon/glucose and build it under `glucose-test`
```bash
mkdir build
cd build
cmake ..
make
```
Grid-search: change variable decay $v_d$ & clause decay $c_d$ in `glucose/core/Solver.cc`. `opt_var_decay`: {0.75, 0.85, 0.95}. `opt_clause_decay`: {0.7, 0.8, 0.9, 0.999}. There is 3 * 5 = 15 settings. For every settings, run `glucose-test/run.sh` to solve `glucose-test/generated`.
Pick the quickist settings of $v_d, c_d$ as the best tuned parameters.
Also, change `generated/real` in `glucose-test/run.sh` to test other set.

Now solve unseened dataset on it, compare the solving time between `real` and `generated` for the tuned parameters.
