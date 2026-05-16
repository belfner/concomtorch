"""Setup script for building concomtorch with CUDA extensions."""

import os
import re
import shutil
import subprocess

import torch
from setuptools import setup


def find_nvcc() -> str | None:
    """
    Locate the nvcc binary, preferring CUDA_HOME over PATH.

    Returns
    -------
    str or None
        Absolute path to nvcc, or None when no nvcc is available.
    """
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home is not None:
        candidate = os.path.join(cuda_home, 'bin', 'nvcc')
        if os.path.exists(candidate):
            return candidate
    return shutil.which('nvcc')


def nvcc_supported_caps() -> list[tuple[int, int]]:
    """
    Query the build toolchain for the compute capabilities it can emit SASS for.

    Runs ``nvcc --list-gpu-code`` and parses the ``sm_XX`` identifiers. The trailing
    digit is the capability minor and the preceding digits the major, per NVIDIA's
    uniform SM naming (``sm_75`` -> (7, 5), ``sm_120`` -> (12, 0)).

    Returns
    -------
    list of tuple of (int, int)
        ``(major, minor)`` capabilities, ascending, deduplicated. Empty when nvcc
        is unavailable.
    """
    nvcc = find_nvcc()
    if nvcc is None:
        return []
    out = subprocess.run(
        [nvcc, '--list-gpu-code'], check=True, capture_output=True, text=True
    ).stdout
    caps = {(int(d[:-1]), int(d[-1])) for d in re.findall(r'sm_(\d+)', out)}
    return sorted(caps)


def resolve_cuda_arch_list() -> str:
    """
    Resolve the dotted TORCH_CUDA_ARCH_LIST for this build.

    An explicit ``TORCH_CUDA_ARCH_LIST`` in the environment is honored verbatim. Otherwise
    the list is derived from the toolchain via :func:`nvcc_supported_caps`, filtered to
    capabilities at or above ``CONCOMTORCH_COMPUTE_MIN`` (default ``7.5``) and sorted
    ascending, so the targeted arches always track the CUDA toolkit actually present.

    Returns
    -------
    str
        Semicolon-separated dotted capabilities, e.g. ``'7.5;8.0;8.6;9.0'``. Empty when
        no toolchain is available and no explicit override is set.
    """
    explicit = os.environ.get('TORCH_CUDA_ARCH_LIST', '').strip()
    if explicit != '':
        return explicit

    floor_str = os.environ.get('CONCOMTORCH_COMPUTE_MIN', '7.5').strip()
    parts = floor_str.split('.')
    floor = (int(parts[0]), int(parts[1]))

    caps = [cc for cc in nvcc_supported_caps() if cc >= floor]
    return ';'.join(f'{maj}.{minor}' for maj, minor in caps)


def gencode_args(arch_list: str) -> list[str]:
    """
    Build the nvcc -gencode flags from a dotted arch list.

    Emits one ``arch=compute_X,code=sm_X`` per capability, then a single PTX fallback
    (``code=compute_X``) for the highest capability so a GPU newer than the build
    toolkit can still JIT.

    Parameters
    ----------
    arch_list : str
        Semicolon-separated dotted capabilities, ascending, e.g. ``'7.5;8.0;9.0'``.

    Returns
    -------
    list of str
        nvcc ``-gencode`` arguments.
    """
    tags = [a.replace('.', '') for a in arch_list.split(';') if a.strip() != '']
    args = [f'-gencode=arch=compute_{t},code=sm_{t}' for t in tags]
    if len(tags) > 0:
        args.append(f'-gencode=arch=compute_{tags[-1]},code=compute_{tags[-1]}')
    return args


cuda_arch_list = resolve_cuda_arch_list()
os.environ['TORCH_CUDA_ARCH_LIST'] = cuda_arch_list

# Match PyTorch's C++11 ABI setting
cxx11_abi = torch._C._GLIBCXX_USE_CXX11_ABI
cxx_abi_flag = f'-D_GLIBCXX_USE_CXX11_ABI={1 if cxx11_abi else 0}'


def build_local_version_suffix() -> str:
    """
    Compose a PEP 440 local version suffix from CONCOMTORCH_CUDA / CONCOMTORCH_TORCH env vars.

    Returns an empty string when either variable is unset, leaving the base version unchanged.
    """
    cuda = os.environ.get('CONCOMTORCH_CUDA', '').strip()
    torch_ver = os.environ.get('CONCOMTORCH_TORCH', '').strip()
    if cuda == '' or torch_ver == '':
        return ''
    return f'+{cuda}torch{torch_ver}'


def build_install_requires() -> list[str]:
    """
    Compose the runtime install_requires list.

    When CONCOMTORCH_TORCH is set (the CI wheel build path), pin the torch dependency to the
    matching minor with ``torch==X.Y.*`` so a single built wheel satisfies any patch release
    of that minor. The env var carries the full build-time patch (e.g. ``2.6.1``); the minor
    is derived here and used for the dependency pin, while the full patch appears in the
    wheel's PEP 440 local version segment via :func:`build_local_version_suffix`. When unset
    (a local source build), leave torch unpinned so the existing environment satisfies the
    dependency.
    """
    torch_ver = os.environ.get('CONCOMTORCH_TORCH', '').strip()
    if torch_ver == '':
        return ['torch']
    parts = torch_ver.split('.')
    if len(parts) < 2:
        raise RuntimeError(
            f"CONCOMTORCH_TORCH must look like 'X.Y' or 'X.Y.Z', got {torch_ver!r}"
        )
    minor = f'{parts[0]}.{parts[1]}'
    return [f'torch=={minor}.*']


def make_cuda_extension():
    """
    Build the CUDAExtension descriptor.

    Imports torch.utils.cpp_extension lazily so the host environment can install or sync
    package metadata without CUDA_HOME present. When CUDA_HOME is unset, return an empty
    extension list so editable installs succeed for Python tooling.
    """
    if os.environ.get('CUDA_HOME') is None and not os.path.exists('/usr/local/cuda'):
        print('setup.py: CUDA_HOME not set; skipping CUDAExtension. Wheel will not contain the compiled op.')
        return []

    from torch.utils.cpp_extension import CUDAExtension

    return [
        CUDAExtension(
            name='concomtorch._C',
            sources=[
                'csrc/ops.cpp',
                'csrc/cuda/ops_cuda.cpp',
                'csrc/cuda/bke_kernels.cu',
            ],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-std=c++17',
                    cxx_abi_flag,
                ],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '-Xcompiler', cxx_abi_flag,
                ]
                + gencode_args(cuda_arch_list),
            },
            py_limited_api=False,
        )
    ]


ext_modules = make_cuda_extension()

cmdclass = {}
if len(ext_modules) > 0:
    from torch.utils.cpp_extension import BuildExtension
    cmdclass['build_ext'] = BuildExtension.with_options(use_ninja=True)

setup(
    name='concomtorch',
    version='0.1.0' + build_local_version_suffix(),
    install_requires=build_install_requires(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
