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
import os
import shlex
import subprocess
import sys
from pathlib import Path

from docker_pool import (
    CUDA_MATRIX,
    build_image,
    image_tag,
    list_resident,
)


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


def torch_minor(version: str) -> str:
    """
    Extract the MAJOR.MINOR portion of a torch version string.
    """
    parts = version.split('.')
    if len(parts) < 2:
        raise ValueError(f'Unexpected torch version: {version!r}')
    return f'{parts[0]}.{parts[1]}'


def run(cmd: list[str], *, env: dict | None = None) -> None:
    print('>>', ' '.join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, env=env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--torch', dest='torch_version', required=True,
                   help='PyTorch version spec, e.g. 2.4.1')
    p.add_argument('--cuda', dest='cuda_variant', required=True,
                   choices=sorted(CUDA_MATRIX.keys()),
                   help='CUDA variant matching PyTorch wheel channel, e.g. cu121')
    p.add_argument('--py', dest='py_abis', nargs='+', required=True,
                   help='One or more CPython ABI tags to build, e.g. cp310 cp311 cp312')
    p.add_argument('--project-dir', dest='project_dir', default='.',
                   help='Path to the project directory.')
    p.add_argument('--output-dir', dest='output_dir', default='wheelhouse',
                   help='Directory on the host to receive built wheels.')
    p.add_argument('--ensure-image', action='store_true',
                   help='Build the docker image if missing (default: error out instead).')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cuda_major_minor, _ = CUDA_MATRIX[args.cuda_variant]

    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
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

    torch_mm = torch_minor(args.torch_version)

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
            f'CONCOMTORCH_TORCH={torch_mm} '
            'AUDITWHEEL_PLAT="manylinux_2_28_x86_64"'
        ),
        'CIBW_BUILD_VERBOSITY': '2',
        'CIBW_CONTAINER_ENGINE': 'docker; create_args: -v concomtorch-pip-cache:/root/.cache/pip',
    })

    run(
        [sys.executable, '-m', 'cibuildwheel', '--platform', 'linux',
         '--output-dir', str(output_dir), str(project_dir)],
        env=cibw_env,
    )

    print('\nWheels written to:', output_dir, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
