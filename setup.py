"""Setup script for building concomtorch with CUDA extensions."""

import os

import torch

# CUDA 13.0+ only supports sm_80 and newer (Ampere, Ada, Hopper)
os.environ['TORCH_CUDA_ARCH_LIST'] = os.environ.get('TORCH_CUDA_ARCH_LIST', '8.0;8.6;8.9;9.0')

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

cuda_arch_list = os.environ['TORCH_CUDA_ARCH_LIST']

# Match PyTorch's C++11 ABI setting
cxx11_abi = torch._C._GLIBCXX_USE_CXX11_ABI
cxx_abi_flag = f'-D_GLIBCXX_USE_CXX11_ABI={1 if cxx11_abi else 0}'

setup(
    name='concomtorch',
    ext_modules=[
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
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)},
)
