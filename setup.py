import os
import sys
from os.path import exists, join
if sys.platform == 'darwin':
    os.environ["CC"] = "/usr/local/opt/llvm/bin/clang++"
    os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang++"
elif sys.platform == 'linux':
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"

from setuptools import setup, Extension, find_packages
from glob import glob
import shutil
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import sys

__version__ = "0.0.1"
module_dir = "./apply"

ext_modules = [
    Extension(
        "apply.omp",
        language='c++',
        sources=[
            "apply/src/omp.cpp", 
            # 'apply/src/operator.h',
            "apply/src/types.cpp"
        ],
        include_dirs=[
            pybind11.get_include(),
            './apply/src'
        ],
        library_dirs=[
            '/usr/local/opt/llvm/lib',
        ],
        libraries=['omp'],
        extra_compile_args=['-std=c++11', '-fopenmp', '-fPIC', '-shared']
        # Example: passing in the version to the compiled code
        # define_macros=[('VERSION_INFO', __version__)],
    )
]

# #############################################
# build
# #############################################
# setup(
#     name="operator_omp",
#     version=__version__,
#     author="Jianbai Ye",
#     author_email="jianbaiye@outlook.com",
#     # url="https://github.com/pybind/python_example",
#     description="Cpp extension support",
#     long_description="",
#     ext_modules=ext_modules,
#     cmdclass={"build_ext": build_ext},
#     zip_safe=False,
# )
# # for ext in ext_modules:
# #     build_ext(ext)
# for ext_file in glob("./*.so"):
#     if exists(join(module_dir, ext_file)):
#         print(f"old {os.path.basename(ext_file)} exists, remove it")
#         os.remove(join(module_dir, ext_file))
#     shutil.move(ext_file, module_dir)

# #############################################
# install
# #############################################

setup(name='apply',
      version=__version__,
      description='A simple autograd implementation based on PyTorch',
      url='https://github.com/gusye1234/PyTrace',
      author="Jianbai Ye",
      author_email='jianbaiye@outlook.com',
      license='MIT',
      packages=['apply'],
      ext_modules=ext_modules,
      cmdclass={"build_ext": build_ext},
      zip_safe=False
)
print("Done")
