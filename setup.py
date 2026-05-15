"""Setup script for building concomtorch with CUDA extensions."""

import os

import torch
from setuptools import setup

# CUDA 13.0+ only supports sm_80 and newer (Ampere, Ada, Hopper)
os.environ['TORCH_CUDA_ARCH_LIST'] = os.environ.get('TORCH_CUDA_ARCH_LIST', '8.0;8.6;8.9;9.0')

cuda_arch_list = os.environ['TORCH_CUDA_ARCH_LIST']

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
                + [
                    f'-gencode=arch=compute_{arch.replace(".", "")},code=sm_{arch.replace(".", "")}'
                    for arch in cuda_arch_list.split(';')
                ],
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
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
