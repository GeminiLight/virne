from config import get_config, show_config, save_config, load_config
from data.generator import Generator
from base import BasicScenario


def run(config):
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")

    print(f'Use {config.solver_name} Solver...\n')
    # Load environment and algorithm
    scenario = BasicScenario.from_config(config)

    # Enable interaction between the environment and the solver 
    scenario.run()

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")


if __name__ == '__main__':
    # Please refer to `base.loader` to obtain all available solvers

    # 1. Get Config
    # The key settings are controlled with config.py
    # while other advanced settings are listed in settings/*.yaml
    config = get_config()

    # You can modify some settings directly here.
    # An example:
    # config.solver_name = 'grc_rank' # modify the algorithm of the solver
    # config.shortest_method = 'mcf'  # modify the shortest path algorithm to Multi-commodity Flow
    # config.num_train_epochs = 100   # modify the number of trainning epochs

    # 2. Generate Dataset
    # Although we do not generate a static dataset,
    # the environment will automatically produce a random dataset.
    p_net, v_net_simulator = Generator.generate_dataset(
        config, 
        p_net=False, 
        v_nets=False, 
        save=False) # Here, no dataset will be generated and saved.

    # 3. Start to Run
    # A scenario with an environment and a solver will be create following provided config.
    # The interaction between the environment and the solver will happen in this scenario.
    run(config)
