# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
from setuptools import setup, find_packages
import os

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, 'verl/version/version')) as f:
    __version__ = f.read().strip()

install_requires = [
  'accelerate==1.12.0',
  'codetiming==1.4.0',
  'datasets==4.4.2',
  'dill==0.4.0',
  'einops==0.8.1',
  'flash_attn==2.7.4.post1',
  'hydra-core==1.3.2',
  'liger_kernel==0.6.4',
  'msgspec==0.20.0',
  'numpy==1.26.4',
  'omegaconf==2.3.0',
  'packaging==25.0',
  'pandas==2.3.3',
  'peft==0.18.0',
  'psutil==7.2.1',
  'pyarrow==22.0.0',
  'pybind11==3.0.1',
  'pylatexenc==2.10',
  'ray[data,train,tune,serve]==2.48.0',
  'safetensors==0.7.0',
  'sentencepiece==0.2.1',
  'tensordict==0.6.2',
  'torch==2.6.0',
  'torchdata==0.11.0',
  'transformers==4.57.3',
  'triton==3.2.0',
  'tqdm==4.67.1',
  'vllm==0.8.3',
  'wandb==0.23.1',
  'xformers==0.0.29.post2',
]

TEST_REQUIRES = ['pytest', 'yapf', 'py-spy']
PRIME_REQUIRES = ['pyext']
GEO_REQUIRES = ['mathruler']

extras_require = {
  'test': TEST_REQUIRES,
  'prime': PRIME_REQUIRES,
  'geo': GEO_REQUIRES,
}

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='verl',
    version=__version__,
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    license='Apache 2.0',
    author='Wei Li, Tao Feng, Hangjie Yuan, Aojun Lu, Yanan Sun',
    description='Official implementation of TARP for retaining stepwise reasoning in sequential reinforcement learning',
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={'': ['version/*'],
                  'verl': ['trainer/config/*.yaml'],},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
