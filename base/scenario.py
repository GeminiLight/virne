import tqdm
import pprint


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

    def run(self, num_epochs=1, start_epoch=0):
        # load pretrained model
        if hasattr(self.solver, 'load_model') and self.config.pretrained_model_path != '':
            self.solver.load_model(self.config.pretrained_model_path)
        # execute pretrain
        if hasattr(self.solver, 'learn') and self.config.num_train_epochs > 0:
            print(f"\n{'-' * 20}  Pretrain  {'-' * 20}\n")
            self.solver.learn(self.env, num_epochs=self.config.num_train_epochs)
            print(f"\n{'-' * 20}    Done    {'-' * 20}\n")

        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            print(f'\nEpoch {epoch_id}') if self.verbose >= 2 else None
            pbar = tqdm.tqdm(desc=f'Running with {self.solver.name} in epoch {epoch_id}', total=self.env.num_vns) if self.verbose <= 1 else None
            instance = self.env.reset()

            while True:
                solution = self.solver.solve(instance)
                next_instance, _, done, info = self.env.step(solution)

                if pbar is not None: 
                    pbar.update(1)
                    pbar.set_postfix({
                        'ac': f'{info["success_count"] / info["vn_count"]:1.2f}',
                        'r2c': f'{info["total_r2c"]:1.2f}',
                        'inservice': f'{info["inservice_count"]:05d}',
                    })

                if done:
                    break
                instance = next_instance
  
            if pbar is not None: pbar.close()
            summary_info = self.env.summary_records()
            if self.verbose == 0:
                pprint.pprint(summary_info)


class TimeWindowScenario(Scenario):

    def __init__(self, env, solver, config):
        super(TimeWindowScenario, self).__init__(env, solver, config)

    def run(self, num_epochs=1, start_epoch=0):
        self.ready()
        print(f"\n{'-' * 20}     Run    {'-' * 20}\n")

        for epoch_id in range(start_epoch, start_epoch + num_epochs):
            print(f'Epoch {epoch_id}') if self.verbose >= 1 else None
            obs = self.env.reset()
            while True:
                action = self.solver.solve(obs)
                next_obs, reward, done, info = self.env.step(action)
                if done:
                    break
                obs = next_obs
                