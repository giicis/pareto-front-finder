#!/usr/bin/python3.7
"""ExactParetoFrontFinder main file."""

import argparse
import dataclasses
import errno
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict

from pyomo.core.base.PyomoModel import Model
from pyomo.environ import SolverFactory, summation
from pyomo.environ import value
from pyomo.opt import OptSolver

import nrp

START = 'start'
DAT_LD = 'data_load'
MD_RUN = 'model_run'
PLOT = 'plot'
SOLVER_THREADS = 16


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class RunResult:
    iteration: int
    profit: float
    cost: float
    x: str
    y: str
    time: float

    @classmethod
    def from_args(self, iteration: int, profit: float, cost: float, x: str, y: str, time: float) -> 'RunResult':
        return RunResult(
            iteration=iteration,
            profit=profit,
            cost=cost,
            x=x,
            y=y,
            time=time
        )


def create_model(data_path: str) -> Model:
    """ Creates a concrete model of the nrp problem
        :param data_path: Where the data file is located
        :return: nrp concrete model
    """
    return nrp.abstract_model().create_instance(data_path)


def fill_model(instance, e_constraint_val: float = None):
    """Fills an abstract  model with e_constraint if e_constraint_val != None
    :param instance: A concrete model
    :param e_constraint_val: Value to profit the cost value, defaults to None
    :param e_constraint_val: int, optional
    :return: Concrete pyomo nrp model
    """

    if e_constraint_val:
        try:
            e_constraint_val = int(e_constraint_val)
            instance.l[1] = summation(instance.cost, instance.x) <= e_constraint_val
        except KeyError:
            instance.l.add(summation(instance.cost, instance.x) <= e_constraint_val)
    return instance


def run(concrete_model: Model, solver: OptSolver, current_iteration: int) -> RunResult:
    """
    """
    solve_kwargs = {}
    if solver.name != 'glpk':
        solve_kwargs['warmstart'] = True

    t0 = time.perf_counter()
    res = solver.solve(concrete_model, **solve_kwargs)
    t1 = time.perf_counter()

    x = ""
    y = ""
    for i in concrete_model.x.values():
        x += str(int(i.value))
    for i in concrete_model.y.values():
        y += str(int(i.value))

    x_as_hex = f'{int(x):x}'
    y_as_hex = f'{int(y):x}'

    profits = value(concrete_model.OBJ)
    costs = value(concrete_model.cost_constraint)

    return RunResult.from_args(
        current_iteration, profits, costs, x_as_hex, y_as_hex, t1 - t0
    )


def write_times(time_struct: dict):
    """Writes all the times
    :param time_struct: Dictionary of times
    :type time_struct: dict
    """

    load_t = time_struct[DAT_LD] - time_struct[START]
    model_run_t = time_struct[MD_RUN] - time_struct[DAT_LD]
    plot_t = time_struct[PLOT] - time_struct[MD_RUN]
    total_t = time_struct[PLOT] - time_struct[START]
    with open('./output/times.txt', 'w+') as file:
        file.write('Data load: ' + ' %s seconds\t %s %% \n' %
                   (round(load_t, 7), round(load_t * 100 / total_t)))
        file.write('Model run: ' + ' %s seconds\t %s %% \n' %
                   (round(model_run_t, 7), round(model_run_t * 100 / total_t)))
        file.write('     Plot: ' + ' %s seconds\t %s %% \n' %
                   (round(plot_t, 7), round(plot_t * 100 / total_t)))
        file.write('    TOTAL: ' + ' %s seconds\t %s %% \n' %
                   (round(total_t, 7), round(total_t * 100 / total_t)))


def model_run(instance, arguments: dict) -> List[RunResult]:
    """Runs the model and retrieves all the gotten results
    :param instance: A concrete pyomo model
    :param arguments: Command-line arguments
    :type arguments: dict
    :return: List of the gotten results (iteration, cost, profit)
    :rtype: list
    """

    def cost_l(x):
        return x > float(arguments['cost']) if arguments['cost'] else lambda x: True

    def profit_l(x):
        return x > float(arguments['profit']
                         ) if arguments['profit'] else lambda x: True

    results = []
    epsilon = float(arguments['epsilon'])
    inform_each = int(arguments['informEach'])

    solver = SolverFactory(arguments['solver'])  # Create solver to use
    solver_supports_threads = solver.name in ['cbc', 'cplex']
    if solver_supports_threads:
        solver.options['threads'] = SOLVER_THREADS

    iteration = 0
    model = fill_model(instance)
    results.append(run(model, solver, iteration))

    iteration += 1
    max_iterations = int(arguments['iterations'])
    while (iteration < max_iterations) \
            and results[-1].profit > 0 \
            and results[-1].cost > 0 \
            and cost_l(results[-1].cost) \
            and profit_l(results[-1].profit):

        model = fill_model(instance, results[-1].cost - epsilon)
        results.append(run(model, solver, iteration))

        if iteration % inform_each == 0:
            print('Iteration {}: Cost = {}, Profit = {}'.format(
                iteration, results[-1].cost, results[-1].profit)
            )
        iteration += 1
    return results


def write_results(results: List[RunResult]):
    try:
        os.makedirs('./output')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    with open('./output/result.json', 'w+') as file:
        json.dump(results, file, cls=EnhancedJSONEncoder)


def main(arguments: dict):
    """Runs the main method in order to get the results and plot them.
    :param arguments: CLI Arguments
    :type arguments: dict
    """
    time_struct = dict()
    time_struct[START] = time.time()
    c_model = create_model(arguments['data_path'])

    time_struct[DAT_LD] = time.time()
    results = model_run(c_model, arguments)
    time_struct[MD_RUN] = time.time()

    time_struct[PLOT] = time.time()

    write_results(results)

    # write_times(time_struct)


def get_parser():
    """Creates parser with it's cli arguments
    :return: Parser object
    :rtype: obj
    """

    parser = argparse.ArgumentParser(description='Finds the Pareto Front of the ' +
                                                 'Requirements-Customers problem.')
    parser.add_argument('--iterations', dest='iterations', default=1000,
                        help='Defines maximum amount of iterations (default: 1000)')
    parser.add_argument('--data', dest='data_path', default='nrp_100c_140r.dat',
                        help='Defines data path from where to get all te model\'s values ' +
                             '(default: \'./nrp_100c_140r.dat\')')
    parser.add_argument('--solver', dest='solver', default='glpk',
                        help='Defines the solver executable path (default: glpk)')
    parser.add_argument('--cost', dest='cost', default=None,
                        help='Defines the limit for the cost objetive')
    parser.add_argument('--profit', dest='profit', default=None,
                        help='Defines the limit for the profit objetive')
    parser.add_argument('--epsilon', dest='epsilon', default='0.001',
                        help='Defines the epsilon value in order to limit/bound certain variable' +
                             '(default: 0.001)')
    parser.add_argument('--informEach', dest='informEach', default='10',
                        help='Defines how often the algorithm reports partial results.' +
                             '(default: 10 iterations)')
    return parser.parse_args()


if __name__ == "__main__":
    main(vars(get_parser()))
