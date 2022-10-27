#!/usr/bin/python3.7
"""ExactParetoFrontFinder main file."""

import os
import re
import errno
import time
import json
import argparse
import numpy as np
import nrp
from pyomo.environ import SolverFactory,  summation

START = 'start'
DAT_LD = 'data_load'
MD_RUN = 'model_run'
PLOT = 'plot'

"""
12/08/2020
    Version 02 del main de pareto front finder
        Modificado:
        - Se utiliza pyomo para la creacion de modelos
        - Se pueden utilizar todos los solvers que soporta pyomo
        - El archivo de ahora es un .dat especificado en: https://pyomo.readthedocs.io/en/stable/working_abstractmodels/data/datfiles.html?highlight=.dat
        - Borrado funciones load_data, format_data, load_model
"""



def create_model(data_path:str):
    """ Creates a concrete model of the nrp problem
        :param data_path: Where the data file is located
        :return: nrp concrete model
    """
    return nrp.abstract_model().create_instance(data_path)



def fill_model(instance,e_constraint_val: float = None):
    """Fills an abstract  model with e_constraint if e_constraint_val != None
    :param instance: A concrete model
    :param e_constraint_val: Value to profit the cost value, defaults to None
    :param e_constraint_val: int, optional
    :return: Concrete pyomo nrp model
    """

    if e_constraint_val:
        try:
            e_constraint_val = int(e_constraint_val)
            instance.l[1] = summation(instance.cost,instance.x) <= e_constraint_val
        except KeyError:
            instance.l.add(summation(instance.cost,instance.x) <= e_constraint_val)
    return instance


def run(concrete_model, solver) -> dict:
    """ Runs the concrete model with the specified solver
    :param concrete_model: A Model already created by pyomo library
    :param solver: A constructed solver to solve the model
    :return: Dictionary representation of the profits and costs
    :rtype: dict
    """
    if solver.name != 'glpk':
        res = solver.solve(concrete_model,warmstart=True)
    else:
        res = solver.solve(concrete_model)
    x = ""
    y = ""
    for i in concrete_model.x.values():
        x += str(int(i.value))
    for i in concrete_model.y.values():
        y += str(int(i.value))

    profits = sum(concrete_model.profit[c] * concrete_model.y[c].value for c in concrete_model.customers)
    costs = sum(concrete_model.cost[r] * concrete_model.x[r].value for r in concrete_model.requierements)
    return {'profit': profits, 'cost': costs, 'x': x, 'y': y}


def plot(x_values: [int], y_values: [int]):
    """Plots the gotten values
    :param x_values: X-axis values
    :type x_values: int[]
    :param y_values: Y-axis values
    :type y_values: int[]
    """

    bokeh_plot.output_file("./output/graphs.html")
    tools = "hover,crosshair,pan,wheel_zoom,box_zoom,reset,tap,save,box_select,poly_select"
    plot_obj = bokeh_plot.figure(title="Gráficos", x_axis_label='Costos', y_axis_label='Ganancias',
                                 tools=tools)
    x_color = np.random.random(size=len(x_values)) * 100
    y_color = np.random.random(size=len(y_values)) * 100
    colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x_color, 30+2*y_color)]
    plot_obj.circle(x_values, y_values, size=10, fill_alpha=0.5, fill_color=colors)
    bokeh_plot.save(plot_obj)


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


def model_run(instance,arguments: dict):
    """Runs the model and retrieves all the gotten results
    :param instance: A concrete pyomo model
    :param arguments: Command-line arguments
    :type arguments: dict
    :return: List of the gotten results (iteration, cost, profit)
    :rtype: list
    """

    result = []
    solver = SolverFactory(arguments['solver']) # Create solver to use
    if solver.name in ['cbc','cplex']:          # If solver support multiple threads
        solver.options['threads'] = 4           # Assing n threads  to solver. Be cautious with overheading
    epsilon = float(arguments['epsilon'])
    data_path = arguments['data_path']

    def cost_l(x): return x > float(arguments['cost']) if arguments['cost'] else lambda x: True
    def profit_l(x): return x > float(arguments['profit']
                                      ) if arguments['profit'] else lambda x: True
    informEach = int(arguments['informEach'])
    def over_zero(x): return x['cost'] > 0 and x['profit'] > 0

    result.append({'iteration': 0, **run(fill_model(instance), solver)})

    
    iteration = 1
    # result[iteration - 1]['cost'] = 500
    while (iteration < int(arguments['iterations'])) and over_zero(result[iteration - 1]) and \
            cost_l(result[iteration - 1]['cost']) and profit_l(result[iteration - 1]['profit']):
        model = fill_model(instance,result[iteration - 1]['cost'] - epsilon)
        result.append({'iteration': iteration, **run(model, solver)})
        if iteration % informEach == 0:
            print('Iteration {}: Cost = {}, Profit = {}'.format(
                iteration, result[iteration - 1]['cost'], result[iteration - 1]['profit']))
        iteration += 1
    return result


def main(arguments: dict):
    """Runs the main method in order to get the results and plot them.
    :param arguments: CLI Arguments
    :type arguments: dict
    """
    print("pid: ",os.getpid())
    time_struct = dict()
    time_struct[START] = time.time()
    c_model = create_model(arguments['data_path'])
    time_struct[DAT_LD] = time.time()
    result = model_run(c_model,arguments)
    time_struct[MD_RUN] = time.time()
    plot(list(map(lambda x: x['cost'], result)), list(map(lambda x: x['profit'], result)))
    time_struct[PLOT] = time.time()
    # creates output folder if it doesn't exists
    try:
        os.makedirs('./output')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    # Writes the results to the final file
    with open('./output/result.json', 'w+') as file:
        json.dump(result, file)

    write_times(time_struct)
    # glpsol --cuts --fpump --mipgap 0.001 --model problem.mod --data problem.dat


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
