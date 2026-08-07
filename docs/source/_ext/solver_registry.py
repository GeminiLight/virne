"""Sphinx directive for rendering Virne's runtime solver registry."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.ext.autodoc.mock import mock


# NumPy remains real because Virne calculates positional embeddings at import time.
# The remaining packages are not needed to inspect class metadata or the registry.
MOCK_IMPORTS = [
    'colorama',
    'colorlog',
    'gym',
    'matplotlib',
    'networkx',
    'ortools',
    'pandas',
    'pyg_lib',
    'sklearn',
    'sympy',
    'torch',
    'torch_cluster',
    'torch_geometric',
    'torch_scatter',
    'torch_sparse',
    'torch_spline_conv',
    'tqdm',
    'wandb',
]

CATEGORY_ORDER = [
    'exact',
    'rounding',
    'heuristic',
    'node_ranking',
    'meta_heuristic',
    'u_learning',
    'r_learning',
]

CATEGORY_LABELS = {
    'exact': 'Exact solvers',
    'rounding': 'Rounding solvers',
    'heuristic': 'Heuristic solvers',
    'node_ranking': 'Node-ranking solvers',
    'meta_heuristic': 'Meta-heuristic solvers',
    'u_learning': 'Unsupervised-learning solvers',
    'r_learning': 'Reinforcement-learning solvers',
}


def _make_row(values: Iterable[str], *, header: bool = False) -> nodes.row:
    row = nodes.row()
    for value in values:
        entry = nodes.entry()
        paragraph = nodes.paragraph()
        if header:
            paragraph += nodes.strong(text=value)
        else:
            paragraph += nodes.literal(text=value)
        entry += paragraph
        row += entry
    return row


def _make_table(entries: list[tuple[str, type]]) -> nodes.table:
    table = nodes.table(classes=['solver-registry'])
    tgroup = nodes.tgroup(cols=2)
    table += tgroup
    tgroup += nodes.colspec(colwidth=35)
    tgroup += nodes.colspec(colwidth=65)

    thead = nodes.thead()
    thead += _make_row(('Command', 'Implementation'), header=True)
    tgroup += thead

    tbody = nodes.tbody()
    for solver_name, solver_class in entries:
        implementation = f'{solver_class.__module__}.{solver_class.__name__}'
        tbody += _make_row((solver_name, implementation))
    tgroup += tbody
    return table


class SolverRegistryDirective(Directive):
    """Render the solvers registered by the current Virne package."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        try:
            with mock(MOCK_IMPORTS):
                from virne.solver import SolverRegistry

            registry = SolverRegistry.list_registered()
        except Exception as exc:  # pragma: no cover - exercised by Sphinx error reporting
            message = self.state_machine.reporter.error(
                f'Unable to load Virne SolverRegistry: {exc}',
                line=self.lineno,
            )
            return [message]

        grouped: dict[str, list[tuple[str, type]]] = defaultdict(list)
        for solver_name, solver_class in registry.items():
            grouped[getattr(solver_class, 'type', 'unknown')].append(
                (solver_name, solver_class)
            )

        result: list[nodes.Node] = []
        categories = CATEGORY_ORDER + sorted(set(grouped) - set(CATEGORY_ORDER))
        for category in categories:
            entries = grouped.get(category)
            if not entries:
                continue
            result.append(nodes.rubric(text=CATEGORY_LABELS.get(category, category)))
            result.append(_make_table(sorted(entries, key=lambda entry: entry[0])))
        return result


def setup(app):
    app.add_directive('solver-registry', SolverRegistryDirective)
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
