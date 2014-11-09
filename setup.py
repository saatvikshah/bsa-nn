from setuptools import setup, find_packages
import sys, os
from pip.req import parse_requirements

version = '0.1'

install_reqs = parse_requirements("requirements.txt")

reqs = [str(ir.req) for ir in install_reqs]

setup(name='bsa',
      version=version,
      description="Implementation of Backtracking Search Algorithm on a Neural Network Architecture supporting upto a single hidden layer",
      long_description="""\
""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='bsa nn bsann ga geneticalgorithm',
      author="Saatvik Shah('tangy')",
      author_email='saatvikshah1994@gmail.com',
      url='',
      license='',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=True,
      install_requires=reqs,
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
