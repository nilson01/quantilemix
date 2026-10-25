"""Build/import/test a local wheel in a fresh disposable Python environment."""
import argparse
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--minimum', action='store_true',
                    help='Use the declared minimum dependency constraints (Python 3.10 in CI)')
args = parser.parse_args()
report_path = root / 'validation/wheel-check.json'
# Remove only this check's generated report so an early failure cannot leave
# a prior successful report looking like the result of the current run.
report_path.unlink(missing_ok=True)
(root / 'validation/pytest-wheel.xml').unlink(missing_ok=True)
if args.minimum and not (3, 10) <= sys.version_info[:2] <= (3, 12):
    parser.error('--minimum needs Python 3.10, 3.11 or 3.12 for the pinned wheels')
constraints = root / 'validation/minimum-constraints.txt'
if args.minimum and not constraints.is_file():
    raise SystemExit('Missing validation/minimum-constraints.txt')
wheel = sorted((root / 'dist').glob('*.whl'))
if len(wheel) != 1:
    raise SystemExit('Expected exactly one built wheel in dist/')
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           VECLIB_MAXIMUM_THREADS='1', PYTHONNOUSERSITE='1',
           PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
env.pop('PYTHONPATH', None)
started = time.perf_counter()
with tempfile.TemporaryDirectory(prefix='qgcomp-wheel-check-') as work:
    work = Path(work)
    py = work / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    install = [str(py), '-m', 'pip', 'install', '--only-binary=:all:']
    if args.minimum:
        install.extend(['--constraint', str(constraints)])
    install.extend([str(wheel[0]), 'pytest>=8'])
    commands = [
        [sys.executable, '-m', 'venv', str(work / 'venv')],
        install,
    ]
    logs = []
    for command in commands:
        completed = subprocess.run(command, cwd=work, env=env, text=True, encoding='utf-8', stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=180)
        logs.append(completed.stdout)
        if completed.returncode:
            (root / 'validation/wheel-install.log').write_text('\n'.join(logs))
            raise SystemExit(completed.returncode)
    shutil.copytree(root / 'tests', work / 'tests', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(root / 'pyproject.toml', work / 'pyproject.toml')
    # Optional-checker unit tests use fake R responses and need only these
    # development scripts. They do not install or execute an R runtime.
    (work / 'validation').mkdir()
    (work / 'validation/server_cox_bootstrap').mkdir()
    shutil.copy2(root / 'validation/server_cox_bootstrap/audit_pilot.py',
                 work / 'validation/server_cox_bootstrap/audit_pilot.py')
    for name in ('audit_study.py', 'study.sbatch'):
        shutil.copy2(root / 'validation/server_cox_bootstrap' / name,
                     work / 'validation/server_cox_bootstrap' / name)
    for name in ('check_installed_r.py', 'export_installed_r.R', 'minimum-constraints.txt',
                 'calibration.py', 'calibration_reference.py', 'calibration_r.R',
                 'calibration-config.json', 'check_calibration_results.py',
                 'check_weighted_r.py', 'export_weighted_r.R',
                 'check_formula_r.py', 'export_formula_r.R',
                 'check_missing_r.py', 'export_missing_r.R',
                 'check_binomial_r.py', 'export_binomial_r.R',
                 'check_poisson_r.py', 'export_poisson_r.R',
                 'check_poisson_offset_r.py', 'export_poisson_offset_r.R',
                 'check_bootstrap_r.py', 'export_bootstrap_r.R',
                 'check_quadratic_r.py', 'export_quadratic_r.R',
                 'check_interactions_r.py', 'export_interactions_r.R',
                 'check_gaussian_ee_r.py', 'export_gaussian_ee_r.R',
                 'check_poisson_ee_r.py', 'export_poisson_ee_r.R',
                 'check_poisson_ee_offset_r.py', 'export_poisson_ee_offset_r.R',
                 'check_binomial_ee_r.py', 'export_binomial_ee_r.R',
                 'check_weighted_ee_r.py', 'export_weighted_ee_r.R',
                 'check_quadratic_gaussian_ee_r.py', 'export_quadratic_gaussian_ee_r.R',
                 'check_risk_ratio_r.py', 'export_risk_ratio_r.R',
                 'check_risk_ratio_multilevel_r.py', 'export_risk_ratio_multilevel_r.R',
                 'check_odds_ratio_r.py', 'export_odds_ratio_r.R',
                 'check_result_tables_r.py', 'export_result_tables_r.R',
                 'check_poisson_bootstrap_r.py', 'export_poisson_bootstrap_r.R',
                 'check_poisson_bootstrap_offset_r.py', 'export_poisson_bootstrap_offset_r.R',
                 'check_covariate_interactions_r.py', 'export_covariate_interactions_r.R',
                 'check_cox_r.py', 'export_cox_r.R',
                 'check_cox_weights_r.py', 'export_cox_weights_r.R',
                 'check_cox_offset_r.py', 'export_cox_offset_r.R',
                 'check_cox_cluster_r.py', 'export_cox_cluster_r.R',
                 'check_cox_ties_r.py', 'export_cox_ties_r.R',
                 'check_cox_curves_r.py', 'export_cox_curves_r.R',
                 'check_cox_bootstrap_r.py', 'export_cox_bootstrap_python.py',
                 'export_cox_bootstrap_stage_r.R',
                 'check_cox_survcurve_r.py', 'export_cox_survcurve_r.R',
                 'cox_bootstrap_preflight.py', 'cox_bootstrap_timing.py',
                 'cox_bootstrap_calibration.py', 'check_cox_bootstrap_calibration.py',
                 'COX_BOOTSTRAP_PROTOCOL.md',
                 'check_tobit_r.py', 'export_tobit_r.R',
                 'check_multinomial_r.py', 'export_multinomial_r.R',
                 'check_zero_inflated_r.py', 'export_zero_inflated_r.R',
                 'check_hurdle_r.py', 'export_hurdle_r.R',
                 'check_covariance_helpers_r.py', 'export_covariance_helpers_r.R',
                 'check_partials_r.py', 'export_partials_r.R',
                 'check_glm_partials_r.py', 'export_glm_partials_r.R',
                 'check_pointwise_noboot_r.py', 'export_pointwise_noboot_r.R',
                 'check_pointwise_boot_r.py', 'export_pointwise_boot_r.R',
                 'check_modelbound_boot_r.py', 'export_modelbound_boot_r.R',
                 'check_modelbound_ee_r.py', 'export_modelbound_ee_r.R',
                 'check_msm_predict_r.py', 'export_msm_predict_r.R',
                 'check_qgcomp_dispatch_r.py', 'export_qgcomp_dispatch_r.R',
                 'paper_reproduction.py', 'check_paper_results.py',
                 'check_paper_r.py', 'export_paper_r.R',
                 'compare_paper_r_versions.py',
                 'paper_scenario5.py', 'check_paper_scenario5.py',
                 'check_paper_scenario5_r.py', 'export_paper_scenario5_r.R',
                 'check_paper_scenario5_cross_platform.py',
                 'binomial_calibration.py', 'check_binomial_calibration.py',
                 'poisson_calibration.py', 'check_poisson_calibration.py',
                 'poisson_offset_calibration.py', 'check_poisson_offset_calibration.py',
                 'poisson_bootstrap_calibration.py', 'check_poisson_bootstrap_calibration.py',
                 'poisson_bootstrap_offset_calibration.py', 'check_poisson_bootstrap_offset_calibration.py',
                 'audit_poisson_bootstrap_offset_summary.py',
                 'risk_ratio_calibration.py', 'check_risk_ratio_calibration.py',
                 'risk_ratio_q4_calibration.py', 'check_risk_ratio_q4_calibration.py',
                 'aggregate_risk_ratio_q4.py',
                 'odds_ratio_q4_calibration.py', 'check_odds_ratio_q4_calibration.py',
                 'aggregate_odds_ratio_q4.py', 'reference_odds_ratio_q4_cases.py',
                 'check_risk_ratio_separation_r.py', 'check_risk_ratio_separation_r.R',
                 'risk-ratio-separation-r-results.json'):
        shutil.copy2(root / 'validation' / name, work / 'validation' / name)
    for name in ('bootstrap_calibration.py', 'check_bootstrap_calibration.py',
                 'bootstrap-calibration-config.json', 'bootstrap-reference-results.json'):
        shutil.copy2(root / 'validation' / name, work / 'validation' / name)
    check = subprocess.run([str(py), '-I', '-c',
        'import quantilemix,sys,json; from pathlib import Path; from importlib.metadata import version; '
        'assert Path(quantilemix.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); '
        'print(json.dumps({"python":sys.version,"platform":sys.platform,"installed_versions":'
        '{name:version(name) for name in ("quantilemix","numpy","scipy","pytest")}}))'],
        cwd=work, env=env, text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    logs.append(check.stdout)
    if check.returncode:
        (root / 'validation/wheel-install.log').write_text('\n'.join(logs))
        raise SystemExit(check.returncode)
    child_environment = json.loads(check.stdout)
    installed_versions = child_environment['installed_versions']
    if args.minimum:
        pins = dict(line.split('==') for line in constraints.read_text().splitlines()
                    if line and not line.startswith('#'))
        for name in ('numpy', 'scipy', 'pytest'):
            if installed_versions[name] != pins[name]:
                raise SystemExit(f'Minimum dependency mismatch: {name}')
    # The Windows source suite took 212 seconds at 1,304 cases in CI. The
    # isolated wheel suite needs room for test-environment overhead as well.
    completed = subprocess.run([str(py), '-I', '-m', 'pytest', '-q', 'tests',
                               f'--junitxml={root / "validation/pytest-wheel.xml"}'],
                              cwd=work, env=env, text=True, encoding='utf-8', stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=360)
    logs.append(completed.stdout)
    (root / 'validation/wheel-install.log').write_text('\n'.join(logs))
    junit = ET.parse(root / 'validation/pytest-wheel.xml').getroot()
    suites = [junit] if junit.tag == 'testsuite' else list(junit.iter('testsuite'))
    counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    if counts['tests'] <= counts['skipped']:
        raise SystemExit('Wheel verification executed no tests')
    if completed.returncode == 0 and (counts['failures'] or counts['errors']):
        raise SystemExit('Wheel test exit code disagrees with the failure report')
    freeze = subprocess.run([str(py), '-m', 'pip', 'freeze'], cwd=work, env=env,
                            text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    if freeze.returncode:
        raise SystemExit(freeze.returncode)
    (root / 'validation/wheel-environment.txt').write_text(freeze.stdout)
    report = {
        'status': 'passed' if completed.returncode == 0 else 'failed',
        'wheel': wheel[0].name, 'returncode': completed.returncode,
        'elapsed_seconds': time.perf_counter()-started,
        'clean_environment': True, 'test_copy_outside_source_tree': True,
        'python': child_environment['python'], 'source_import_path_excluded': True,
        'platform': child_environment['platform'], 'dependency_profile': 'minimum' if args.minimum else 'latest',
        'installed_versions': installed_versions,
        'test_counts': counts,
        'wheel_sha256': hashlib.sha256(wheel[0].read_bytes()).hexdigest(),
        'constraints_sha256': hashlib.sha256(constraints.read_bytes()).hexdigest() if args.minimum else None,
    }
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    print(completed.stdout)
    raise SystemExit(completed.returncode)
