
Basic Usage
===========


Minimal Example
---------------


.. code:: python

    import os
    from config import get_config, show_config, save_config, load_config
    from data import Generator
    from base import BasicScenario
    from solver import REGISTRY

    # 1. Get Config
    # The key settings are controlled with config.py
    # while other advanced settings are listed in settings/*.yaml
    config = get_config()

    # generate p_net and v_net dataset
    Generator.generate_dataset(config)


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
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")
    # load Env and Solver
    solver_info = REGISTRY.get(config.solver_name)
    Env, Solver = solver_info['env'], solver_info['solver']
    print(f'Use {config.solver_name} Solver (Type = {solver_info["type"]})...\n')

    # create scenario with Env and Solver
    scenario = BasicScenario.from_config(Env, Solver, config)
    scenario.run()

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")
