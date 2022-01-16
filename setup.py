from setuptools import setup, find_packages
setup(name='lslkit',
      version='0.1',
      description='API for PyLSL Development and Experiments',
      author='Morgan Stuart',
      packages=find_packages(),
      requires=['pylsl', 'numpy', 'pandas',
                'attrs', 'tqdm', 'scipy'])