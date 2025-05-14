# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch


class RolloutBuffer:
    
    def __init__(self, ):
        self.curr_idx = 0
        self.basic_items = ['observations', 'actions', 'rewards', 'dones', 'next_observations', 'logprobs', 'values']
        self.calc_items = ['advantages', 'returns']
        self.extend_items = ['hidden_states', 'cell_states', 'action_mask', 'entropies']
        self.reset()

    @property
    def all_items(self):
        return self.basic_items + self.calc_items + self.extend_items

    def reset(self):
        self.curr_idx = 0
        for item in self.all_items:
            setattr(self, item, [])
        # for item in self.all_items:
        #     item_list = getattr(self, item)
        #     del item_list[:]

    def clear(self):
        self.reset()

    def size(self):
        return len(self.logprobs)

    def add(self, obs, action, raward, done, logprob, value=None, next_obs=None):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(raward)
        self.dones.append(done)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.next_observations.append(next_obs)
        self.curr_idx += 1
        if hasattr(self, 'use_prioritized') and self.use_prioritized:
            transition = (obs, action, raward, next_obs, done)
            self.prioritized_buffer.add(transition)
    
    def get_subbuffer(self, indices):
        sub_buffer = RolloutBuffer()
        for item in self.all_items:
            item_list = getattr(self, item)
            sub_item_list = getattr(sub_buffer, item)
            if len(item_list) == 0:
                continue
            for idx in indices:
                sub_item_list.append(item_list[idx])
        return sub_buffer

    def extend(self, item):
        self.extend_items += [item]
        setattr(self, item, [])

    def split_with_ratios(self, ratios=[0.5, 0.5]):
        assert sum(ratios) == 1
        buffer_size = self.size()
        indices = list(range(buffer_size))
        len_sub_buffers = len(ratios)
        sub_buffer_sizes = [int(buffer_size * ratio) for ratio in ratios]
        sub_buffer_sizes[-1] = buffer_size - sum(sub_buffer_sizes[:-1])
        sub_buffer_indices = []
        for i in range(len_sub_buffers):
            sub_buffer_indices.append(indices[:sub_buffer_sizes[i]])
            del indices[:sub_buffer_sizes[i]]
        sub_buffers = []
        for i in range(len_sub_buffers):
            sub_buffers.append(self.get_subbuffer(sub_buffer_indices[i]))
        return sub_buffers

    def split_with_instance(self, ):
        buffer_size = self.size()
        indices = list(range(buffer_size))
        sub_buffer_indices = []
        for i in range(buffer_size):
            if self.dones[i]:
                sub_buffer_indices.append(indices[:i+1])
                del indices[:i+1]
        sub_buffers = []
        for i in range(len(sub_buffer_indices)):
            sub_buffers.append(self.get_subbuffer(sub_buffer_indices[i]))
        return sub_buffers

    def merge(self, buffer):
        for item in self.all_items:
            main_item_list = getattr(self, item)
            sub_item_list = getattr(buffer, item)
            main_item_list += sub_item_list

    def compute_returns_and_advantages(self, last_value=0., gamma=0.99, gae_lambda=0.98, method='gae', values=None) -> None:
        """
        Compute expected return and advantage for each step.

        Args:
            last_value: value of last step
            gamma: discount factor
            gae_lambda: lambda factor for Generalized Advantage Estimator
            method: method to compute advantage, one of ['gae', 'mc', 'td']
        """
        assert method in ['gae', 'mc', 'td']
        # calculate expected return (Genralized Advantage Estimator)
        if isinstance(last_value, torch.Tensor): last_value = last_value.item()
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
        elif method == 'mc':
            self.returns = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.rewards), reversed(self.dones)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (gamma * discounted_reward)
                self.returns.insert(0, discounted_reward)
        elif method == 'td':
            self.returns = []
            for step in reversed(range(buffer_size)):
                if step == buffer_size - 1:
                    next_values = last_value
                else:
                    next_values = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
                discounted_reward = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
                self.returns.insert(0, discounted_reward)
        if method == 'ar_gae':
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


class RolloutBufferWithCost(RolloutBuffer):

    def __init__(self, ):
        self.safe_rl_items = ['costs', 'cost_values', 'cost_returns', 'baseline_cost_returns']
        super(RolloutBufferWithCost, self).__init__()

    @property
    def all_items(self):
        return self.basic_items + self.calc_items + self.extend_items + self.safe_rl_items

    def compute_cost_returns(self, gamma=1., method='reachability'):
        assert method in ['reachability', 'cumulative']
        assert len(self.costs) == len(self.rewards)
        if method == 'reachability':
            state_cost = 0
            for cost, is_terminal in zip(reversed(self.costs), reversed(self.dones)):
                if is_terminal:
                    state_cost = -float('inf')
                state_cost = max(cost, (gamma * state_cost))
                self.cost_returns.insert(0, state_cost)
        if method == 'cumulative':
            self.costs = [c if c >= 0 else 0. for c in self.costs]
            state_cost = 0
            for cost, is_terminal in zip(reversed(self.costs), reversed(self.dones)):
                if is_terminal:
                    state_cost = 0
                state_cost = cost + (gamma * state_cost)
                self.cost_returns.insert(0, state_cost)


class RolloutBufferWithAvgReturn(RolloutBuffer):

    def __init__(self, ):
        super(RolloutBufferWithAvgReturn, self).__init__()

    def compute_returns_and_advantages(self, last_value=0., gamma=0.99, gae_lambda=0.98, method='gae') -> None:
        """
        Compute expected average return and advantage for each step.

        Args:
            last_value: value of last step
            gamma: discount factor
            gae_lambda: lambda factor for Generalized Advantage Estimator
            method: method to compute advantage, one of ['gae', 'mc']
        """
        assert method in ['gae', 'td']
        if isinstance(last_value, torch.Tensor): last_value = last_value.item()
        buffer_size = self.size()
        self.returns = [0] * buffer_size
        self.advantages = [0] * buffer_size
        if method == 'gae':
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
        elif method == 'td':
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


def compute_returns_with_gae(items, values, dones, last_value=0., gamma=0.99, gae_lambda=0.98) -> None:
    """
    Compute expected item (rewards or costs) return and advantage for each step.

    Args:
        last_value: value of last step
        gamma: discount factor
        gae_lambda: lambda factor for Generalized Advantage Estimator
    """
    if isinstance(last_value, torch.Tensor): last_value = last_value.item()
    buffer_size = len(items)
    returns = [0] * buffer_size
    advantages = [0] * buffer_size
    last_gae_lam = 0
    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_values = last_value
        else:
            next_values = values[step + 1]
        next_non_terminal = 1.0 - dones[step]
        delta = items[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[step] = last_gae_lam
        returns[step] = advantages[step] + values[step]
    return returns, advantages


def compute_returns_with_mc(items, dones, gamma=0.99) -> None:
    """
    Compute expected item (rewards or costs) return and advantage for each step.

    Args:
        last_value: value of last step
        gamma: discount factor
        gae_lambda: lambda factor for Generalized Advantage Estimator
    """
    buffer_size = len(items)
    returns = []
    discounted_item = 0
    for item, is_terminal in zip(reversed(items), reversed(dones)):
        if is_terminal:
            discounted_item = 0
        discounted_item = item + (gamma * discounted_item)
        returns.insert(0, discounted_item)
    return returns


def compute_returns_with_td(items, values, dones, last_value=0., gamma=0.99) -> None:
    """
    Compute expected item (rewards or costs) return and advantage for each step.

    Args:
        last_value: value of last step
        gamma: discount factor
        gae_lambda: lambda factor for Generalized Advantage Estimator
    """
    if isinstance(last_value, torch.Tensor): last_value = last_value.item()
    buffer_size = len(items)
    returns = []
    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_values = last_value
        else:
            next_values = values[step + 1]
        next_non_terminal = 1.0 - dones[step]
        discounted_item = items[step] + gamma * next_values * next_non_terminal - values[step]
        returns.insert(0, discounted_item)
    return returns

# Optionally, add a PrioritizedReplayBuffer for Rainbow DQN
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, transition):
        max_prio = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = np.array(self.priorities)
        else:
            prios = np.array(self.priorities[:len(self.buffer)])
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

if __name__ == '__main__':
    buffer = RolloutBuffer(1, 1)
    
    temp = [1, 2, 3]
    for i in range(10):
        buffer.temp = temp
        buffer.observations.append(buffer.temp)
        temp.append(i)

    print(buffer.observations)
