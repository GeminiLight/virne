from algo import Agent, Environment, BatchEnv, environment
from config import get_config

def run(config):
    if config.run_mode == 'test':
        config.num_epochs = 1
    for i in range(config.num_epochs):
        batch_env = BatchEnv(**vars(config))
        agent = Agent(**vars(config))
        agent.run(batch_env, run_mode=config.run_mode)
        agent.save_model(config.save_dir)
        batch_env.batch[0].save_records(config.records_dir, f'/{config.run_mode}_i_records.csv')


if __name__ == '__main__':
    config, _ = get_config()
    config.run_mode = 'test'
    config.batch_size = 2
    run(config)
    