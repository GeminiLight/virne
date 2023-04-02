# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import inspect

from base import SolutionStepEnvironment

class Registry:
    """
    A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """
    def __init__(self, name):
        self._name = name
        self._solver_dict = {}

    def __repr__(self):
        return (
            f'{self.__class__.__name__ }(name={self._name}, items={list(self._solver_dict.keys())})'
        )

    @property
    def name(self):
        """Return the name of the registry."""
        return self._name

    @property
    def solver_dict(self) -> dict:
        """Return a dict mapping names to classes."""
        return self._solver_dict

    def get(self, solver_name: str) -> dict:
        """Get the class that has been registered under the given key."""
        solver_name = solver_name.lower()
        if solver_name not in self._solver_dict.keys():
            raise KeyError(f'The solver {solver_name} is not in the {self.name} registry')
        return self._solver_dict.get(solver_name)

    def _add(
            self, 
            solver_name: str, 
            solver_cls: object, 
            env_cls: object = SolutionStepEnvironment, 
            solver_type: str = 'unknown'
        ) -> None:
        """
        Register a module class, and add it to the registry.
        """
        if not inspect.isclass(solver_cls) and not inspect.isclass(env_cls):
            raise TypeError(f'module must be a class, but got {type(solver_cls)} and {type(env_cls)}')
        if solver_name in self._solver_dict:
            raise KeyError(f'{solver_name} is already registered in {self.name}')
        self._solver_dict[solver_name.lower()] = {'solver': solver_cls, 'env': env_cls, 'type': solver_type.lower()}

    def register(self, solver_name: str, env_cls: object = SolutionStepEnvironment, solver_type: str = 'unknown') -> object:
        """
        Register a solver class with a solver name.

        Args:
            solver_name (str): The name of the solver.
            env_cls (class): The environment class.
            solver_type (str): The type of the solver.
        """
        def _register(solver_cls):
            solver_cls.name = solver_name
            solver_cls.type = solver_type
            self._add(solver_name, solver_cls, env_cls, solver_type)
            return solver_cls
        return _register


REGISTRY = Registry('Virne')


register = REGISTRY.register
get = REGISTRY.get
