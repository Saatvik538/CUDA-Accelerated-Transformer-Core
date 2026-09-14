from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

cuda_extension_modules = []
if CUDA_HOME is not None:
    cuda_extension_modules.append(
        CUDAExtension(
            name="cuda_transformer_core._cuda_attention",
            sources=[
                "cuda_transformer_core/csrc/attention.cpp",
                "cuda_transformer_core/csrc/attention_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    )

setup(
    name="cuda_transformer_core",
    version="0.1.0",
    packages=["cuda_transformer_core"],
    ext_modules=cuda_extension_modules,
    cmdclass={"build_ext": BuildExtension},
)
