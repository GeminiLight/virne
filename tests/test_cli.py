import subprocess
import sys
import tomllib
from importlib import resources
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_module_cli(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - executes the current test interpreter
        [sys.executable, '-m', 'virne', *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def test_default_configs_are_package_resources() -> None:
    config_root = resources.files('virne').joinpath('configs')

    assert config_root.joinpath('main.yaml').is_file()
    assert config_root.joinpath('learning.yaml').is_file()
    assert config_root.joinpath('p_net_setting', 'default.yaml').is_file()
    assert config_root.joinpath('v_sim_setting', 'default.yaml').is_file()


def test_console_script_points_to_cli() -> None:
    with (PROJECT_ROOT / 'pyproject.toml').open('rb') as file:
        project_metadata = tomllib.load(file)

    assert project_metadata['project']['scripts']['virne'] == 'virne.cli:main'


def test_top_level_public_exports_remain_available() -> None:
    from virne import Generator, SolverRegistry

    assert Generator.__name__ == 'Generator'
    assert SolverRegistry.__name__ == 'SolverRegistry'


def test_module_cli_loads_default_config_outside_repository(tmp_path: Path) -> None:
    result = run_module_cli('--cfg', 'job', cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert 'solver_name: ppo_dual_gat+' in result.stdout
    assert 'num_v_nets: 1000' in result.stdout


def test_module_cli_reports_version(tmp_path: Path) -> None:
    result = run_module_cli('--version', cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'virne 1.0.0'


def test_legacy_main_script_remains_compatible(tmp_path: Path) -> None:
    result = subprocess.run(  # noqa: S603 - executes the checked-in compatibility script
        [sys.executable, str(PROJECT_ROOT / 'main.py'), '--cfg', 'job'],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert 'solver_name: ppo_dual_gat+' in result.stdout
