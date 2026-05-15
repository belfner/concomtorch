#!/usr/bin/env python3
"""
Build manylinux wheels for one (torch_version, cuda_variant, py_abis) combination.

Builds a manylinux_2_28 docker image with the matching CUDA toolkit, then runs cibuildwheel
inside it to produce wheels in the output directory. Wheels are named with a PEP 440 local
version suffix `+cu{N}torch{X.Y}` via CONCOMTORCH_CUDA and CONCOMTORCH_TORCH env vars consumed
by setup.py.
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

CUDA_MATRIX = {
    'cu118': ('11.8', '11-8'),
    'cu121': ('12.1', '12-1'),
    'cu124': ('12.4', '12-4'),
    'cu126': ('12.6', '12-6'),
    'cu127': ('12.7', '12-7'),
    'cu128': ('12.8', '12-8'),
    'cu129': ('12.9', '12-9'),
    'cu130': ('13.0', '13-0'),
}

DEFAULT_MANYLINUX_IMAGE = 'quay.io/pypa/manylinux_2_28_x86_64'


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


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print('>>', ' '.join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def torch_minor(version: str) -> str:
    """
    Extract the MAJOR.MINOR portion of a torch version string.

    Local version suffixes embed the minor release (e.g. '+cu121torch2.4') because every patch
    of a given torch minor shares the same ABI.

    Parameters
    ----------
    version : str
        Full torch version string, e.g. '2.4.1'.

    Returns
    -------
    str
        Major.minor, e.g. '2.4'.
    """
    parts = version.split('.')
    if len(parts) < 2:
        raise ValueError(f'Unexpected torch version: {version!r}')
    return f'{parts[0]}.{parts[1]}'


def make_dockerfile_text(
    manylinux_image: str,
    cuda_major_minor: str,
    cuda_pkg_suffix: str,
) -> str:
    """
    Render a Dockerfile that installs CUDA toolkit + GCC toolset 12 on manylinux_2_28.
    """
    return f"""\
# syntax=docker/dockerfile:1
ARG BASE_IMAGE={manylinux_image}
FROM ${{BASE_IMAGE}}

RUN yum -y install dnf-plugins-core curl git && \\
    yum -y config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \\
    yum -y clean all && rm -rf /var/cache/yum

RUN yum -y install gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ gcc-toolset-12-binutils && \\
    yum -y clean all && rm -rf /var/cache/yum

RUN yum -y install cuda-toolkit-{cuda_pkg_suffix} && \\
    yum -y clean all && rm -rf /var/cache/yum

ENV CUDA_HOME=/usr/local/cuda-{cuda_major_minor}
ENV PATH=$CUDA_HOME/bin:$PATH

ENV PATH=/opt/rh/gcc-toolset-12/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:/opt/rh/gcc-toolset-12/root/usr/lib:$LD_LIBRARY_PATH
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Build manylinux wheels inside a CUDA-enabled manylinux container.'
    )
    p.add_argument('--torch', dest='torch_version', required=True,
                   help='PyTorch version spec, e.g. 2.4.1')
    p.add_argument('--cuda', dest='cuda_variant', required=True,
                   choices=sorted(CUDA_MATRIX.keys()),
                   help='CUDA variant matching PyTorch wheel channel, e.g. cu121')
    p.add_argument('--py', dest='py_abis', nargs='+', required=True,
                   help='One or more CPython ABI tags to build, e.g. cp310 cp311 cp312')
    p.add_argument('--image-tag', dest='image_tag', default=None,
                   help='Tag to assign to the built Docker image (default auto-generated).')
    p.add_argument('--manylinux-image', dest='manylinux_image', default=DEFAULT_MANYLINUX_IMAGE,
                   help=f'Base image (default: {DEFAULT_MANYLINUX_IMAGE})')
    p.add_argument('--project-dir', dest='project_dir', default='.',
                   help='Path to the project directory (defaults to current directory).')
    p.add_argument('--output-dir', dest='output_dir', default='wheelhouse',
                   help='Directory on the host to receive built wheels.')
    return p.parse_args()


def compute_cibw_build_pattern(py_abis: list[str]) -> str:
    abis = normalize_abis(py_abis)
    parts = [f'{abi}-manylinux_x86_64' for abi in abis]
    return ' '.join(parts)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    if args.cuda_variant not in CUDA_MATRIX:
        print(f'Unsupported CUDA variant: {args.cuda_variant}', file=sys.stderr)
        sys.exit(2)
    cuda_major_minor, cuda_pkg_suffix = CUDA_MATRIX[args.cuda_variant]

    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dirs(output_dir)

    image_tag = (
        args.image_tag
        or f'concomtorch-manylinux-{args.cuda_variant}:torch-{args.torch_version}'
    )

    dockerfile_text = make_dockerfile_text(
        manylinux_image=args.manylinux_image,
        cuda_major_minor=cuda_major_minor,
        cuda_pkg_suffix=cuda_pkg_suffix,
    )

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        dockerfile_path = tmpdir / 'Dockerfile'
        dockerfile_path.write_text(dockerfile_text, encoding='utf-8')

        run([
            'docker', 'build',
            '-t', image_tag,
            '--build-arg', f'BASE_IMAGE={args.manylinux_image}',
            str(tmpdir),
        ])

    norm_abis = normalize_abis(args.py_abis)
    torch_mm = torch_minor(args.torch_version)

    cibw_env = os.environ.copy()
    cibw_env.update({
        'CIBW_MANYLINUX_X86_64_IMAGE': image_tag,
        'CIBW_BUILD': compute_cibw_build_pattern(norm_abis),
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

    print('\nWheels written to:', output_dir)


if __name__ == '__main__':
    main()
