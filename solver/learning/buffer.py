import copy

from jinja2 import pass_eval_context


class RolloutBuffer:
    
    def __init__(self, max_size=None):
        self.curr_idx = 0
        self.max_size = max_size

        self.basic_items = ['observations', 'actions', 'rewards', 'dones', 'next_observations', 'logprobs', 'values']
        self.calc_items = ['advantages', 'returns']
        self.extend_items = ['hiddens', 'action_masks']
        self.safe_items = ['costs', 'cost_returns']
        self.all_items = self.basic_items + self.calc_items + self.extend_items + self.safe_items

        self.observations = []
        self.next_observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []

        self.advantages = []
        self.returns = []
        self.entropies = []
        self.action_masks = []

        self.hiddens = []
        # for safe RL
        self.costs = []
        self.cost_returns = []

    
    def reset(self):
        self.curr_idx = 0
        for item in self.all_items:
            item_list = getattr(self, item)
            del item_list[:]

    def clear(self):
        self.reset()

    def size(self):
        return len(self.logprobs)

    def is_full(self):
        if self.max_size is None:
            return False
        return self.curr_id == self.max_size

    def add(self, obs, action, raward, done, logprob, value=None):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(raward)
        self.dones.append(done)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.curr_idx += 1
    
    def merge(self, buffer):
        for item in self.all_items:
            main_item_list = getattr(self, item)
            sub_item_list = getattr(buffer, item)
            main_item_list += sub_item_list
        # self.observations += copy.deepcopy(buffer.observation)
        # self.actions += copy.deepcopy(buffer.actions)
        # self.rewards += copy.deepcopy(buffer.rewards)
        # self.dones += copy.deepcopy(buffer.dones)
        # self.logprobs += copy.deepcopy(buffer.logprobs)
        # self.values += copy.deepcopy(buffer.values)
        # self.advantages += copy.deepcopy(buffer.advantages)
        # self.returns += copy.deepcopy(buffer.returns)
        # self.hiddens += copy.deepcopy(buffer.hiddens)

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.98, method='gae') -> None:
        # calculate expected return (Genralized Advantage Estimator)
        buffer_size = self.size()
        self.returns = [0] * buffer_size
        self.advantages = [0] * buffer_size

        if method == 'gae':
            last_gae_lam = 0
            for step in reversed(range(buffer_size)):
                if step == buffer_size - 1:
                    next_values = last_value
                else:
                    next_values = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
                delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
                last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
                self.returns[step] = self.advantages[step] + self.values[step]
        elif method == 'ar_td':
            self.dones[-1] = False
            mean_reward = sum(self.rewards) / len(self.rewards)
            for step in reversed(range(buffer_size)):
                if step == buffer_size - 1:
                    next_values = last_value
                else:
                    next_values = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
                self.advantages[step] = self.rewards[step] - mean_reward + next_values * next_non_terminal - self.values[step]
                self.returns[step] = self.advantages[step] + self.values[step]
        elif method == 'ar_gae':
            self.dones[-1] = False
            last_gae_lam = 0
            mean_reward = sum(self.rewards) / len(self.rewards)
            for i in range(len(self.rewards)):
                self.rewards[i] = self.rewards[i] - mean_reward
            # for step in reversed(range(buffer_size)):
            #     if step == buffer_size - 1:
            #         next_values = last_value
            #     else:
            #         next_values = self.values[step + 1]
            #     next_non_terminal = 1.0 - self.dones[step]
            #     delta = self.rewards[step] + next_values * next_non_terminal - self.values[step] - mean_reward
            #     last_gae_lam = delta + gae_lambda * next_non_terminal * last_gae_lam
            #     self.advantages[step] = last_gae_lam
            #     self.returns[step] = self.advantages[step] + self.values[step]
                # self.returns[step] = self.rewards[step] + next_values - mean_reward
                # print(self.rewards[step], mean_reward, last_gae_lam, delta)
            for step in reversed(range(buffer_size)):
                if step == buffer_size - 1:
                    next_values = last_value
                else:
                    next_values = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
                delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
                last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam
                self.returns[step] = self.advantages[step] + self.values[step]

        elif method == 'mc':
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.rewards), reversed(self.dones)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (gamma * discounted_reward)
                self.returns.insert(0, discounted_reward)

    def compute_mc_returns(self, gamma=0.99):
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            self.returns.insert(0, discounted_reward)

        if len(self.costs) != len(self.rewards):
            return

        discounted_cost = 0
        for cost, is_terminal in zip(reversed(self.costs), reversed(self.dones)):
            if is_terminal:
                discounted_cost = 0
            discounted_cost = cost + (gamma * discounted_cost)
            self.cost_returns.insert(0, discounted_cost)


if __name__ == '__main__':
    buffer = RolloutBuffer(1, 1)
    
    temp = [1, 2, 3]
    for i in range(10):
        buffer.temp = temp
        buffer.observations.append(buffer.temp)
        temp.append(i)

    print(buffer.observations)
