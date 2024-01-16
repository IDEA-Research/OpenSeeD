import setuptools
from setuptools import find_packages
from setuptools.command.install import install
import subprocess
import sys
import re

# # groundingdino needs torch to be installed before it can be installed
# # this is a hack but couldn't find any other way to make it work
# try:
#     import torch
# except:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", 'torch'])
# subprocess.call("python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html", shell=True)

class InstallLocalPackage(install):
    def run(self):
        install.run(self)
        subprocess.call(
            "python openseed/body/encoder/ops/setup.py build install --user", shell=True
        )

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().split('\n')

setuptools.setup(
    name="OpenSeeD", 
    version="0.1.0",
    author="Zhang, Hao and Li, Feng and Zou, Xueyan and Liu, Shilong and Li, Chunyuan and Gao, Jianfeng and Yang, Jianwei and Zhang, Lei",
    author_email="{hzhangcx, fliay}@connect.ust.hk",
    description="A Simple Framework for Open-Vocabulary Segmentation and Detection",
    url="https://github.com/parallel-domain/OpenSeeD",
    install_requires=install_requires,
    packages=find_packages(),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort", "twine", "pytest", "wheel"],
    },
    python_requires=">=3.8",
    cmdclass={'install':InstallLocalPackage},
)
