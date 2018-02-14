import os
from setuptools import setup, find_packages


    def read(fname):
        try:
            return open(os.path.join(os.path.dirname(__file__), fname)).read()
        except:
            return 'Please see: https://github.com/UBC-MDS/ezmodel'

    setup(
        name='ezmodel',
        version='v0.1',
        author=['Alex Kleefeldt', 'Sean Conley', 'Tyler Roberts'],
        author_email='',
        description='Simple modelling diagnostic and workflow tools for use with sklearn',
        long_description=read('README.md'),
        license='',
        keywords='machine learning',
        url='https://github.com/UBC-MDS/ezmodel',
        packages=find_packages(),
        install_requires=['os', 'setuptools', 'scipy','numpy', 'sklearn', 'matplotlib'],
        classifiers=[],
        include_package_data=True
    )
