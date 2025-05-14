import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Type

class BaseActorCritic(nn.Module):

    def __init__(self, *args, **kwargs):
        super(BaseActorCritic, self).__init__()
    
    def act(self, x):
        return self.actor(x)
    
    def evaluate(self, x):
        if not hasattr(self, 'critic'):
            return None
        return self.critic(x)


class ActorCriticWithSharedEncoderBase(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ActorCriticWithSharedEncoderBase, self).__init__()
    
    def act(self, x):
        x = self.encoder(x)
        return self.actor(x)
    
    def evaluate(self, x):
        x = self.encoder(x)
        if not hasattr(self, 'critic'):
            return None
        return self.critic(x)




class ActorCriticRegistry:
    """
    Registry for actor-critic classes. Supports registration and retrieval by name.
    """
    name: str = 'ActorCriticRegistry'
    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, actor_critic_name: str):
        def decorator(handler_cls: Type[nn.Module]):
            if actor_critic_name in cls._registry:
                raise ValueError(f"Actor-Critic '{actor_critic_name}' is already registered.")
            cls._registry[actor_critic_name] = handler_cls
            return handler_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        if name not in cls._registry:
            raise NotImplementedError(f"Actor-Critic '{name}' is not implemented.")
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, Type[nn.Module]]:
        return dict(cls._registry)
