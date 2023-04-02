from .tester import Tester

class SolverTester(Tester):

    def __init__(self, solver):
        super(SolverTester, self).__init__()
        self.solver = solver

    def test_solver(self):
        self.test_solver_init()
        self.test_solver_solve()

    def test_solver_init(self):
        self.solver.init()

    def test_solver_solve(self):
        pass