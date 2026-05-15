#!/usr/bin/env python3
"""
Run cibuildwheel against a pre-built manylinux+CUDA image to produce wheels for one
(torch_version, cuda_variant, py_abis) combination.

Image lifecycle (build, prune) lives in ci/docker_pool.py; this script assumes the image exists.
Wheels emerge named with the PEP 440 local version `+cu{N}torch{X.Y}` because setup.py reads
CONCOMTORCH_CUDA / CONCOMTORCH_TORCH from CIBW_ENVIRONMENT.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import shlex
import subprocess
import sys
from pathlib import Path

from docker_pool import (
    build_image,
    cuda_tag_parts,
    image_tag,
    list_resident,
)

# cibuildwheel resolves test-sources against its own process working directory
# (cibuildwheel/platforms/linux.py passes Path.cwd() into copy_test_sources),
# not --project-dir. The real cibuildwheel subprocess and the preflight query
# are launched with cwd=REPO_ROOT so CIBW_TEST_SOURCES_LINUX=tests deterministically
# copies the repo-root tests/ directory regardless of where this script is invoked.
REPO_ROOT = Path(__file__).resolve().parent.parent


def normalize_abi(tag: str) -> str:
    """
    Accept 'cp310', '310', 'cp39', '39' and return 'cp310' / 'cp39'.
    """
    t = tag.lower().strip()
    if t.startswith('cp'):
        return t
    if t.isdigit():
        return f'cp{t}'
    raise ValueError(f'Unrecognized Python ABI: {tag}')


def normalize_abis(tags: list[str]) -> list[str]:
    return [normalize_abi(t) for t in tags]


def compute_cibw_build_pattern(py_abis: list[str]) -> str:
    """
    Render the CIBW_BUILD glob pattern for the requested CPython ABIs on linux x86_64.
    """
    return ' '.join(f'{abi}-manylinux_x86_64' for abi in normalize_abis(py_abis))


def run(cmd: list[str], *, env: dict | None = None, cwd: Path | None = None) -> None:
    print('>>', ' '.join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, env=env, cwd=cwd)


def preflight_buildable(
    project_dir: Path,
    py_abis: list[str],
    env: dict,
    cwd: Path | None = None,
) -> list[str]:
    """
    Ask cibuildwheel which build identifiers it would emit and report missing ABIs.

    The running cibuildwheel decides the set of Python ABIs it can target. When the
    pinned cibuildwheel is too old (or too new) for a requested ABI it silently
    produces no wheel for it. This queries `--print-build-identifiers` against the
    fully composed CIBW environment and returns the requested ABIs that cibuildwheel
    would not build.

    Parameters
    ----------
    project_dir : Path
        Path to the project directory passed to cibuildwheel.
    py_abis : list[str]
        Requested CPython ABI tags (any accepted form, e.g. 'cp310', '310').
    env : dict
        The fully composed CIBW environment used for the real build.
    cwd : Path, optional
        Working directory for the cibuildwheel subprocess, by default None.
        Must match the cwd of the real build so the preflight query reflects
        the same test-sources resolution.

    Returns
    -------
    list[str]
        Normalized ABI tags (e.g. 'cp314') that cibuildwheel would not build.
    """
    result = subprocess.run(
        [sys.executable, '-m', 'cibuildwheel', '--platform', 'linux',
         '--print-build-identifiers', str(project_dir)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    identifiers = result.stdout.split()
    missing = [
        abi for abi in normalize_abis(py_abis)
        if not any(ident.startswith(f'{abi}-') for ident in identifiers)
    ]
    return missing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--torch', dest='torch_version', required=True,
                   help='PyTorch version spec, e.g. 2.4.1')
    p.add_argument('--cuda', dest='cuda_variant', required=True,
                   help='CUDA variant matching PyTorch wheel channel, e.g. cu121')
    p.add_argument('--py', dest='py_abis', nargs='+', required=True,
                   help='One or more CPython ABI tags to build, e.g. cp310 cp311 cp312')
    p.add_argument('--compute-min', dest='compute_min', default='7.5',
                   help='Lowest CUDA compute capability (SM) setup.py emits device code for, '
                        'e.g. 7.5. Passed to the build as CONCOMTORCH_COMPUTE_MIN.')
    p.add_argument('--project-dir', dest='project_dir', default='.',
                   help='Path to the project directory.')
    p.add_argument('--output-dir', dest='output_dir', default='wheelhouse',
                   help='Directory on the host to receive built wheels.')
    p.add_argument('--ensure-image', action='store_true',
                   help='Build the docker image if missing (default: error out instead).')
    p.add_argument('--skip-tests', dest='skip_tests', action='store_true',
                   help='Omit the in-container pytest gate and GPU passthrough, for a '
                        'deliberate GPU-less local build. Refused unless --output-dir is '
                        'an explicitly non-"wheelhouse" scratch directory: a skipped-test '
                        'wheel is unverified and must never be indistinguishable from a '
                        'verified artifact. Not exposed via ci/run.py.')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cuda_major_minor, _ = cuda_tag_parts(args.cuda_variant)

    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.skip_tests and output_dir.name == 'wheelhouse':
        print(
            '--skip-tests refuses to write into a "wheelhouse" directory. A '
            'skipped-test wheel is unverified; pass --output-dir to an explicit '
            'scratch directory (e.g. scratch-wheelhouse) so it cannot be mistaken '
            'for a verified artifact or picked up by the publish path.',
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    tag = image_tag(args.cuda_variant)
    resident = {info.cuda_variant for info in list_resident()}
    if args.cuda_variant not in resident:
        if args.ensure_image:
            print(f'Image {tag} not found; building.', flush=True)
            build_image(args.cuda_variant)
        else:
            print(
                f'Image {tag} not resident. Either run ci/docker_pool.py ensure {args.cuda_variant} '
                f'first or pass --ensure-image.',
                file=sys.stderr,
            )
            return 2

    cibw_env = os.environ.copy()
    cibw_env.update({
        'CIBW_MANYLINUX_X86_64_IMAGE': tag,
        'CIBW_BUILD': compute_cibw_build_pattern(args.py_abis),
        'CIBW_BUILD_FRONTEND': 'pip; args: --no-build-isolation',
        'CIBW_SKIP': '*musllinux*',
        'CIBW_BEFORE_BUILD': (
            'rm -rf .venv venv build dist *.egg-info wheelhouse || true && '
            'python -m pip install --upgrade pip && '
            f'python -m pip install "torch=={args.torch_version}+{args.cuda_variant}" '
            f'--index-url https://download.pytorch.org/whl/{args.cuda_variant}/ && '
            'python -m pip install setuptools>=70.1.0 wheel ninja numpy'
        ),
        'CIBW_REPAIR_WHEEL_COMMAND_LINUX': (
            'auditwheel repair -w {dest_dir} {wheel} '
            '--exclude libtorch.so '
            '--exclude libtorch_cpu.so '
            '--exclude libtorch_cuda.so '
            '--exclude libtorch_python.so '
            '--exclude libc10.so '
            '--exclude libc10_cuda.so '
            '--exclude libtorch_global_deps.so '
            '--exclude libcaffe2_nvrtc.so '
            '--exclude libtorch_cuda_linalg.so '
            '--exclude libshm.so'
        ),
        'CIBW_ENVIRONMENT': (
            f'CUDA_HOME=/usr/local/cuda-{cuda_major_minor} '
            f'LD_LIBRARY_PATH=/usr/local/cuda-{cuda_major_minor}/lib64:$LD_LIBRARY_PATH '
            f'CONCOMTORCH_CUDA={args.cuda_variant} '
            f'CONCOMTORCH_TORCH={args.torch_version} '
            f'CONCOMTORCH_COMPUTE_MIN={args.compute_min} '
            'AUDITWHEEL_PLAT="manylinux_2_28_x86_64"'
        ),
        'CIBW_BUILD_VERBOSITY': '2',
    })

    pip_cache_mount = '-v concomtorch-pip-cache:/root/.cache/pip'
    if args.skip_tests:
        # GPU-less local build: no test gate, no GPU passthrough so the
        # container can be created on a host without the NVIDIA runtime.
        cibw_env['CIBW_CONTAINER_ENGINE'] = f'docker; create_args: {pip_cache_mount}'
    else:
        # The build, repair, and test steps reuse one container created from
        # these create_args, so --gpus=all is what gives the cibuildwheel test
        # step GPU visibility for the CUDA-only kernels. The single-token
        # --gpus=all form avoids create_args token-pair splitting.
        cibw_env['CIBW_CONTAINER_ENGINE'] = (
            f'docker; create_args: --gpus=all {pip_cache_mount}'
        )
        cibw_env.update({
            # Runs in the fresh test venv before the wheel install. The wheel
            # pins torch==X.Y.* (setup.py); PyPI does not carry the +cuXXX
            # local-version build, so install the matching CUDA torch from the
            # pytorch channel here. cibuildwheel does not pass --upgrade on the
            # subsequent repaired-wheel install, so this torch is kept.
            'CIBW_BEFORE_TEST_LINUX': (
                f'python -m pip install "torch=={args.torch_version}+{args.cuda_variant}" '
                f'--index-url https://download.pytorch.org/whl/{args.cuda_variant}/'
            ),
            # Runs from cibuildwheel's temporary test cwd against the copied
            # tests/ tree (see CIBW_TEST_SOURCES_LINUX), not the source tree.
            'CIBW_TEST_COMMAND_LINUX': 'python -m pytest -v tests',
            # Copied into the temporary test cwd. Resolves against the
            # cibuildwheel process cwd, which is pinned to REPO_ROOT below.
            'CIBW_TEST_SOURCES_LINUX': 'tests',
            # Installs the repaired wheel as concomtorch[test]; the test extra
            # is declared in package/pyproject.toml. torch is deliberately not
            # in that extra (handled by CIBW_BEFORE_TEST_LINUX above).
            'CIBW_TEST_EXTRAS_LINUX': 'test',
            # conftest.py hard-fails when CONCOMTORCH_REQUIRE_GPU=1 and CUDA is
            # unavailable, so a mis-provisioned runner cannot silently pass the
            # gate with zero GPU tests.
            'CIBW_TEST_ENVIRONMENT_LINUX': (
                f'CONCOMTORCH_REQUIRE_GPU=1 '
                f'CONCOMTORCH_EXPECTED_CUDA={args.cuda_variant} '
                f'CONCOMTORCH_EXPECTED_TORCH={args.torch_version}'
            ),
        })

    missing = preflight_buildable(project_dir, args.py_abis, cibw_env, cwd=REPO_ROOT)
    if len(missing) > 0:
        cibw_version = importlib.metadata.version('cibuildwheel')
        print(
            f'cibuildwheel {cibw_version} will not build these requested ABIs: '
            f'{" ".join(missing)}. The pinned cibuildwheel cannot target them; '
            f'adjust the cibuildwheel pin in the repo-root pyproject.toml or the '
            f'--py request.',
            file=sys.stderr,
        )
        return 3

    run(
        [sys.executable, '-m', 'cibuildwheel', '--platform', 'linux',
         '--output-dir', str(output_dir), str(project_dir)],
        env=cibw_env,
        cwd=REPO_ROOT,
    )

    print('\nWheels written to:', output_dir, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
