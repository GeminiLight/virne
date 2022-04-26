from config import get_config, show_config, save_config, load_config
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

    if config.verbose >= 2: show_config(config)
    if config.if_save_config: save_config(config)
    return scenario

def run(config):
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")

    print(f'Use {config.solver_name} Solver...\n')
    
    # Load environment and algorithm
    scenario = create_scenario(config)
    scenario.run(num_epochs=config.num_epochs, start_epoch=config.start_epoch)

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")


if __name__ == '__main__':
    ## -- available solver -- ##
    # You can find all available solvers in utils.py

    # 1. Get config / Load config
    config = get_config()

    ## ------ Soft update config ------ ###
    # select solver with its name
    config.solver_name = 'pg_cnn'
    config.num_epochs = 1
    config.num_train_epochs = 100
    config.verbose = 1
    config.if_save_records = True
    config.if_temp_save_records = True
    config.pretrained_model_path = ''
    config.summary_file_name = f'summary.csv'
    ## ------         End        ------ ###

    # 2. Generate Dataset
    # Note:
    #   If the dataset does not exist, please generate it before running the solver
    #   If a dataset with the same settings already exists, the dataset will be overwritten 
    pn, vn_simulator = Generator.generate_dataset(config, pn=True, vns=True, save=True)

    # 3. Run with solver
    run(config)
