import itertools
from .solver import Solver
from .exact import *
from .heuristic import *
from .meta_heuristic import *
from .learning import *
from .registry import REGISTRY, register, get

from . import exact, heuristic, meta_heuristic, learning


SOLVERS = {
    'exact': tuple(exact.__all__),
    'heuristic': tuple(heuristic.__all__),
    'meta_heuristic': tuple(meta_heuristic.__all__),
    'learning': tuple(learning.__all__),
}
SOLVERS['all'] = tuple(itertools.chain.from_iterable(SOLVERS.values()))

__all__ = list(SOLVERS['all']) + [
    'Solver',
    'REGISTRY',
    'register',
    'get',
]