from args import get_config
from shortest_path.env import SPEnv
from shortest_path.agent import SPGAT


if __name__ == '__main__':
    config = get_config()
    # env = SPEnv(config)
    env, agent = SPGAT.from_config(config)
    total_timesteps = 5000000
    save_timestep = 500000
    for i in range(int(total_timesteps / save_timestep)):

        agent.learn(total_timesteps=save_timestep)
        agent.save(f'sp-{i}')
    
    # obs = env.reset()
    # epoch_reward = 0

    # while True:
    #     mask = env.generate_action_mask()
    #     action, _states = agent.predict(obs, action_masks=mask, deterministic=True)
    #     next_obs, reward, done, info = env.step(action)

    #     epoch_reward += reward
    #     obs = next_obs

    #     if done: break

    # print(f'cumulative reward in test: {epoch_reward}')

