# os.chdir(os.path.join(os.getcwd(), 'code/virne-dev'))
import os

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from virne.system import BaseSystem

from virne.utils.config import add_simulation_into_config, generate_run_id



@hydra.main(config_path="settings", config_name="main")
def run(config):
    print(f"\n{'-' * 20}    Start     {'-' * 20}\n")
    # ---- Before running the simulation ---- #
    # Configure Customization
    # Method 1. Update the config file
    # Method 2. Use hydra CLI
    # Method 3. Modify the config here [Not recommended]
    if config.experiment.run_id == 'auto':
        config.experiment.run_id = generate_run_id()
    add_simulation_into_config(config)
    # --------------------------------------- #
    system = BaseSystem.from_config(config)
    system.run()

    print(f"\n{'-' * 20}   Complete   {'-' * 20}\n")


if __name__ == '__main__':
    run()
