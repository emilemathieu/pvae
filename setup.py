from setuptools import setup, find_packages
import os

with open('requirements.txt') as fp:
        install_requires = fp.read()

setup(name='pvae',
      version='0.2',
      install_requires=install_requires,
      description='Pytorch implementation of Poincaré Variational Auto-Encoders',
      long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
      url='https://github.com/emilemathieu/pvae',
      author_email='emile.mathieu@stats.ox.ac.uk',
      license='MIT',
      packages=find_packages(),
      zip_safe=False)