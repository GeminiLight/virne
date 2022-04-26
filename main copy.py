# os.chdir(os.path.join(os.getcwd(), 'code/virne-dev'))
import os
from config import get_config
from utils import load_solver
from data.generator import Generator
from base import BasicScenario


def create_scenario(config):
    # load env and solver
    Env, Solver = load_solver(config.solver_name)
    # Create env and solver
    env = Env.from_config(config)
    solver = Solver.from_config(config)
    # Create scenario
    scenario = BasicScenario(env, solver, config)
    return scenario

def main(config):
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")

    print(f'Use {config.solver_name} Solver...')
    
    # Load environment anda lgorithm
    scenario = create_scenario(config)

    scenario.run(num_epochs=config.num_epochs, start_epoch=config.start_epoch)

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")


if __name__ == '__main__':
    config = get_config()
    # Note:
    #   If the dataset does not exist, please generate it before running the solver
    #   If a dataset with the same settings already exists, the dataset will be overwritten 
    Generator.generate_dataset(config, pn=True, vns=True)
    # config.summary_file_name = f'summary_{config.solver_name}.csv'
    main(config)
