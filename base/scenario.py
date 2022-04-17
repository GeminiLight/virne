class Scenario:

    def __init__(self, env, solver, config):
        self.env = env
        self.solver = solver
        self.config = config
        self.verbose = config.verbose


class BasicScenario(Scenario):

    def __init__(self, env, solver, config):
        super(BasicScenario, self).__init__(env, solver, config)

    def reset(self):
        pass

    def ready(self):
        if self.config.num_train_epochs > 0:
            print(f"\n{'-' * 20}  Pretrain  {'-' * 20}\n")
            self.solver.learn(self.env, num_epochs=self.config.num_train_epochs, batch_size=self.config.batch_size)
            print(f"\n{'-' * 20}    Done    {'-' * 20}\n")

    def run(self, num_epochs=1, start_epoch=0):
        self.ready()

        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            print(f'Epoch {epoch_id}') if self.verbose >= 1 else None
            obs = self.env.reset()
            while True:
                action = self.solver.solve(obs)
                next_obs, reward, done, info = self.env.step(action)
                if done:
                    break
                obs = next_obs


class TimeWindowScenario(Scenario):

    def __init__(self, env, solver, config):
        super(TimeWindowScenario, self).__init__(env, solver, config)

    def run(self, num_epochs=1, start_epoch=0):
        self.ready()

        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            print(f'Epoch {epoch_id}') if self.verbose >= 1 else None
            obs = self.env.reset()
            while True:
                action = self.solver.solve(obs)
                next_obs, reward, done, info = self.env.step(action)
                if done:
                    break
                obs = next_obs