# Pareto frond finder

Program to find pareto front of Next release planning problem using epslin-restrictions algorithm

It can be used with the datasets inside the root folder.

To run it 

```bash
python main.py --solver cbc --iterations 200 --workers 16 --data nrp_100c_140r.dat
```
Parameters:
- `--solver` Solver to use. Can be anything soported by [pyomo](http://www.pyomo.org/)
- `--data` data file path
- `--iterations` max amount of iterations to use per solver
- `--informEach` How often the program print partial results
- `--workers` Amount of workers to use

## How this works

1. Calculte profit using a very big number as a cost restriction.
2. Using the result in `1` get the max cost as
3. Split `[0; max_cost]` in `n` partitions. `n` is the number of workers
4. Assign each worker a partition
5. For each worker
    1. Find the pareto front of that partition using the epslins restrictions algorithm

Of course this is a simplified explanation. For a complete descrition, check the source code.

## Results
This program produces a json file with a vector of information for each iteration. You can see how the information is presented in the file `pareto_front_dfisplay.ipynb`