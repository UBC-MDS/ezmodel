from setuptools import setup, find_packages

setup(
    name='ezmodel',
    version='v0.1',
    author=['Alex Kleefeldt', 'Sean Conley', 'Tyler Roberts'],
    author_email='',
    description='Simple modelling diagnostic and workflow tools for use with sklearn',
    long_description='README.md',
    license='',
    keywords='machine learning',
    url='https://github.com/UBC-MDS/ezmodel',
    packages=find_packages(),
    install_requires=['sys', 'setuptools', 'scipy','numpy', 'sklearn', 'matplotlib', 'pandas'],
    classifiers=[],
    include_package_data=True
)
