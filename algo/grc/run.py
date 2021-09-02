from env import GRCEnv
from grc import GRC

from config import get_config

def run(config):
    grc = GRC(**vars(config))
    env = GRCEnv(**vars(config))
    grc.run(env)
    env.save_records(config.records_dir, f'/grc_records.csv')

if __name__ == '__main__':
    config, _  = get_config()

    # GRC Configurations 
    # config.para = default  # [optional]

    # config.reused_vnf = False # [True]

    run(config)