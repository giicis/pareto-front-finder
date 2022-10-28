#!/usr/bin/python3.7
"""ExactParetoFrontFinder main file."""

import argparse
import dataclasses
import errno
import json
import os
import time
from dataclasses import dataclass
from typing import List

from pyomo.core.base.PyomoModel import Model
from pyomo.environ import SolverFactory, summation
from pyomo.environ import value
from pyomo.opt import OptSolver

from nrp import NrpModel

START = 'start'
DAT_LD = 'data_load'
MD_RUN = 'model_run'
PLOT = 'plot'
SOLVER_THREADS = os.cpu_count()


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
    def from_args(cls, iteration: int, profit: float, cost: float, x: str, y: str, elapsed: float) -> 'RunResult':
        return RunResult(
            iteration=iteration,
            profit=profit,
            cost=cost,
            x=x,
            y=y,
            time=elapsed
        )


def run(concrete_model: Model, solver: OptSolver, current_iteration: int) -> RunResult:
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


def find_pareto_front(arguments: dict) -> List[RunResult]:
    def cost_l(x):
        return x > float(arguments['cost']) if arguments['cost'] else lambda x: True

    def profit_l(x):
        return x > float(arguments['profit']
                         ) if arguments['profit'] else lambda x: True

    results = []
    epsilon = float(arguments['epsilon'])
    inform_each = int(arguments['informEach'])
    model = NrpModel(arguments['data_path'])

    solver = SolverFactory(arguments['solver'])  # Create solver to use
    solver_supports_threads = solver.name in ['cbc', 'cplex']
    if solver_supports_threads:
        solver.options['threads'] = SOLVER_THREADS

    iteration = 0
    results.append(run(model.model, solver, iteration))

    iteration += 1
    max_iterations = int(arguments['iterations'])
    while (iteration < max_iterations) \
            and results[-1].profit > 0 \
            and results[-1].cost > 0 \
            and cost_l(results[-1].cost) \
            and profit_l(results[-1].profit):

        model.update_cost_constraint(results[-1].cost - epsilon)
        results.append(run(model.model, solver, iteration))

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
    results = find_pareto_front(arguments)
    write_results(results)


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
