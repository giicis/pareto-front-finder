import dataclasses
import json
from unittest import TestCase

from pyomo.opt import SolverFactory

from main import create_model, run, RunResult


class MainUnitTest(TestCase):

    def setUp(self) -> None:
        data_path = 'nrp_100c_140r.dat'
        solver_name = 'cbc'
        self.concrete_model = create_model(data_path)
        self.solver = SolverFactory(solver_name)

    def test_run_smoke(self):
        run(self.concrete_model, self.solver)

    def test_run_result_to_json_smoke(self):
        result = RunResult(
            1, 1, 1, '1', '1', 1
        )
        json.dumps(dataclasses.asdict(result))
