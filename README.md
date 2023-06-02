# HardSATGEN: Understanding the Difficulty of Hard SAT Formula Generation and A Strong Structure-Hardness-Aware Baseline

Official implementation of SIGKDD 2023 paper "HardSATGEN: Understanding the Difficulty of Hard SAT Formula Generation and A Strong Structure-Hardness-Aware Baseline".

## Acknowledgment

This implementation is based on [G2SAT](https://github.com/JiaxuanYou/G2SAT) [1] and [SAT_generators](https://github.com/i4vk/SAT_generators) [2]. 

## Installation

- Basic environment (tested on)
  - python==3.8
  - networkx==3.0
  - pytorch==1.13.1+cu116
  - torch_geometric
  - pygmtools==0.3.5
  - tensorboardX


- Open source tools
  - [cadical](https://github.com/arminbiere/cadical) [3]: `./postprocess/cadical`
    - build: `./configure && make`
  - [drat-trim](https://github.com/marijnheule/drat-trim) [4]: `./postprocess/drat-trim`
    - build: `make`
  - glucose: See more in "Eval and solver tuning" section.

## Dataset preparation

example dataset: `${dataset_name}`

- formula：put in `./dataset/${dataset_name}/`, named `${formula_name}.cnf` for each formula
- core：put in `./dataset/${dataset_name}_core/`, named `${formula_name}_core` for each
- lcg_stats.csv：exists in `./dataset/`, add each formula information with format as `${dataset_name}/${formula_name}, ${num_variable}, ${num_clause}`

How to build dataset when only formulas in hand:

1. Prepare core: make sure to build cadical & drat-trim first

    ```bash
    bash scripts/solve_core.sh dataset/${dataset_name}
    ```

    Core files will generated under `dataset/${dataset_name}`, move them into `dataset/${dataset_name}_core`.
    Also, the cadical solving log are stored in dataset/class_name.log, check solving time if needed.

2. Prepare csv

    ```bash
    dataset_name/formula_name, num_var, num_clause
    ```

    The information must be correspounding to original cnf file.
    If a cnf file can't be found in csv, then it will not be use to train/test.
    ** The cnf-lcg converting process may cause "Stats not match", model will automatically skip these cnfs. Since the reason are still not clarified, if encountered this in train/test, we recommend you replace the bad files with some files that didn't cause this problem.

## Example Run

1. Train

    ```bash
    # Do not remain unsat core
    python src/main_train.py --epoch_num 201 --data_name ${dataset_name} --model GCN # SAGE; GCN
    # remain unsat core && introduce scrambling operation
    python src/main_train.py --epoch_num 201 --data_name ${dataset_name} --core_flag --model GCN # SAGE; GCN
    ```

    After this step, trained models will be saved in `model/` directory.

2. Use to generate Formulas

    ```bash
    python src/main_test.py --epoch_load 200 --data_name ${dataset_name} --model GCN # SAGE; GCN
    # remain unsat core && introduce scrambling operation
    python src/main_test.py --epoch_num 200 --data_name ${dataset_name} --core_flag --model GCN # SAGE; GCN
    ```

    After this step, generated graphs will be saved to `graphs/` directory. 1 graph is generated out of 1 template by default, check `args.py` for more options.

    Graphs will be saved in a single `.dat` file containing all the generated graphs.

    (It may take fairly long time: Running is fast, but updating networkx takes the majority of time in current implementation.)

    We can then generate CNF formulas from the generated graphs

    ```bash
    python src/eval/conversion_lcg.py --src graphs/${dataset_name}_lcg_GCN_coreTrue_alpha.dat --store-dir formulas/${dataset_name} --action=lcg2sat
    ```

3. Post-processing for Formulas

    ```bash
    cd scripts && bash auto_rmcore.sh ${dataset_name} 200 2255
    # ${1}: class name in ./formulas/train_set
    # ${2}: maximum re-post times
    # ${3}: expected solving time threshold in sec
    
    # multi-task version:
    cd scripts && bash auto_rmcore_multi.sh ${dataset_name} 200 2255
    ```

    The post-processed formulas will be stored in `./formulas/${dataset_name}_post`, logs in `./postprocess/cadical/build`


## Eval and solver tuning

1. Evaluate graph properties of formulas
    Make sure to build this first: 
    
    ```bash
    g++ -o src/eval/scalefree eval/scalefree.cpp
    ```
    Evaluate generated formulas with:
    ```bash
    # evaluate
    python src/eval/evaluate_graphs_lcg.py -s src/eval/scalefree -d formulas/${dataset_name}/ -o ${dataset_name}.csv
    ```
    
1. Solver tuning
    Here we use [glucose](https://github.com/wadoon/glucose) [5] to test the tuning ability. (following the experiments settings of G2SAT)
    Build glucose first. Download and build it under `./glucose`

    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

    Grid-search: change variable decay $v_d$ & clause decay $c_d$ in `glucose/core/Solver.cc`. `opt_var_decay`: {0.75, 0.85, 0.95}. `opt_clause_decay`: {0.7, 0.8, 0.9, 0.999}. There is 3 * 5 = 15 settings. For every settings, run `script/glucose_test.sh` to solve formulas in `glucose/generated`.
    Pick the quickist settings of $v_d, c_d$ as the best tuned parameters.
    Also, change `generated/real` in `glucose-test/run.sh` to test other set.

    Now solve unseen dataset on it, compare the solving time between `real` and `generated` for the tuned parameters.

## References

[1] Jiaxuan You, HaozeWu, Clark Barrett, Raghuram Ramanujan, and Jure Leskovec. 2019. G2SAT: Learning to generate sat formulas. Advances in neural information processing systems 32 (2019).

[2] Iván Garzón, Pablo Mesejo, and Jesús Giráldez-Cru. 2022. On the Performance of Deep Generative Models of Realistic SAT Instances. In 25th International Conference on Theory and Applications of Satisfiability Testing (SAT 2022).

[3] Armin Biere Katalin Fazekas Mathias Fleury and Maximilian Heisinger. 2020. CaDiCaL, kissat, paracooba, plingeling and treengeling entering the SAT competition 2020. SAT COMPETITION 50 (2020), 2020.

[4] Nathan Wetzler, Marijn JH Heule, and Warren A Hunt. 2014. DRAT-trim: Efficient checking and trimming using expressive clausal proofs. In Theory and Applications of Satisfiability Testing–SAT 2014: 17th International Conference, Held as Part of the Vienna Summer of Logic, VSL 2014, Vienna, Austria, July 14-17, 2014. Proceedings 17. Springer, 422–429.

[5] Gilles Audemard and Laurent Simon. 2009. Glucose: a solver that predicts learnt
clauses quality. SAT Competition (2009), 7–8.
