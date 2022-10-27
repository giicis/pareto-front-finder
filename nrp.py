from __future__ import division
from pyomo.environ import NonNegativeIntegers, AbstractModel, Param, RangeSet, Set, Binary, Var, summation, Objective,Constraint, maximize, ConstraintList, SolverFactory
import sys


def abstract_model():
    """
    Creates an abstract model for nrp problem
    """

    # Abstract model definition

    # creating the actual model
    nrp = AbstractModel()

    # Creation of parameters
    nrp.number_of_requierements = Param(within=NonNegativeIntegers)
    nrp.number_of_costumers = Param(within=NonNegativeIntegers)
    nrp.max_cost = Param(within=NonNegativeIntegers, mutable=True)

    # Creation of sets
    nrp.requierements = RangeSet(1, nrp.number_of_requierements)
    nrp.customers = RangeSet(1, nrp.number_of_costumers)
    nrp.profit = Param(nrp.customers)  # Profit of each customer
    nrp.cost = Param(nrp.requierements)  # Cost of requierement
    # (i,j) requierement i should be implemented if j is implemented
    nrp.prerequisite = Set(within=nrp.requierements * nrp.requierements)
    # (i,k) this relation exists if stakeholder k has interest on requierement i
    nrp.interest = Set(within=nrp.customers * nrp.requierements)

    # Creation of variables
    # = 1 if requierement i is implemented in the next release, otherwise 0
    nrp.x = Var(nrp.requierements, domain=Binary)
    # = 1 if all  stakeholder i's interests are satisfied in the next release, otherwise 0
    nrp.y = Var(nrp.customers, domain=Binary)

    # Objetive function

    def obj_expression(nrp):
        return summation(nrp.profit, nrp.y) - summation(nrp.cost, nrp.x) # maximize profit

    nrp.OBJ = Objective(rule=obj_expression, sense=maximize)

    # Defintion of cost constraint rule
    def cost_constraint_rule(nrp):
        return summation(nrp.cost, nrp.x) <= nrp.max_cost

    nrp.cost_constraint = Constraint(rule=cost_constraint_rule)

    # Defition of precedence constraint
    def precedence_constraint_rule(nrp, i, j):
        return nrp.x[i] >= nrp.x[j]
    nrp.precedence_constraint = Constraint(nrp.prerequisite, rule=precedence_constraint_rule)

    # Definition of interest constraint
    # Each tuple in nrp.dat.interest is inverted, so the constraint is also inverted
    def interest_constraint_rule(nrp, i, k):
        return nrp.y[i] <= nrp.x[k]
    nrp.interest_constraint = Constraint(nrp.interest, rule=interest_constraint_rule)
    nrp.l = ConstraintList()
    return nrp


def Main():
    """
    Run the nrp problem and print output
    This main function is used primarly for testing reasons
    first CLI argument is solver. Can be glpk,cbc,cplex or any other solver soported by pyomo
    second CLI argument is the data file to fill the model.
    """

    solver_name = sys.argv[1]
    data_file = sys.argv[2]

    solver = SolverFactory(solver_name)
    nrp = abstract_model().create_instance(data_file)

    res = solver.solve(nrp)
    profits = sum(nrp.profit[c] * nrp.y[c].value for c in nrp.customers)
    costs = sum(nrp.cost[r] * nrp.x[r].value for r in nrp.requierements)
   

    res.write()

    print("Profit: ",profits)
    print("Costs: ",costs)


if __name__ == "__main__":
    Main()
