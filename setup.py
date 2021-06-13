from setuptools import setup, find_packages
setup(name='lslkit',
      version='0.1',
      description='API For LSL Development and Experiments',
      author='Morgan Stuart',
      #packages=['mmz'],
      packages=find_packages(),
      #modules=['feature_processing', 'torch_models'],
      requires=['pylsl', 'numpy', 'pandas',
                #'sklearn', 'torch', 'torchvision',
                'attrs'])