"""Exercise CMD parsing, launcher exit codes and scheduled logging on Windows."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows CMD launchers')
ROOT = Path(__file__).resolve().parents[1]


def environment():
    env = os.environ.copy()
    env.update(MOEX_PYTHON=sys.executable, MOEX_NO_PAUSE='1', PYTHONIOENCODING='utf-8')
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    return env


def run(command, cwd, env):
    return subprocess.run([os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', command],
                          cwd=cwd, env=env, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=30)


def test_update_launcher_reaches_cli_help():
    result = run('update_data.bat --help', ROOT, environment())
    assert result.returncode == 0, result.stdout + result.stderr
    assert '--history-init' in result.stdout
    assert 'unexpected' not in result.stderr


def test_missing_interpreter_reports_error_without_cmd_parse_failure(tmp_path):
    env = environment()
    env['MOEX_PYTHON'] = str(tmp_path / 'missing python.exe')
    result = run('update_data.bat --help', ROOT, env)
    assert result.returncode == 1
    assert 'cannot import moexutils/polars/duckdb' in result.stdout
    assert 'unexpected' not in result.stderr


def test_scheduled_wrapper_logs_and_preserves_failure_code(tmp_path):
    folder = tmp_path / 'launcher with spaces'
    folder.mkdir()
    for name in ('update_data.bat', 'scheduled_update.cmd'):
        shutil.copyfile(ROOT / name, folder / name)
    (folder / 'update_data.py').write_text(
        'import sys\nprint("ARGS:", sys.argv[1:])\nsys.exit(7)\n', encoding='ascii')
    result = run('scheduled_update.cmd --check', folder, environment())
    assert result.returncode == 7, result.stdout + result.stderr
    log = (folder / 'logs' / 'update.log').read_text(encoding='utf-8')
    assert '===== start ' in log and '===== exit 7 ' in log
    assert "ARGS: ['--check']" in log
    assert 'unexpected' not in log
