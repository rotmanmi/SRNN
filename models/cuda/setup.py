from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='srnn',
    ext_modules=[
        CUDAExtension('srnn', [
            'srnn.cpp',
            'srnn_kernels.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    })
