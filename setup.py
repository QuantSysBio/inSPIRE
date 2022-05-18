
from setuptools import setup

setup(
    name='inspire',
    version=0.1,
    description='Helping to integrate Spectral Predictors and Rescoring.',
    author='John Cormican',
    author_email='john.cormican@mpinat.mpg.de',
    packages=[
        'inspire',
        'inspire.input',
    ],
    long_description=open('README.md').read(),
    py_modules=[
        'inspire',
        'inspire.input',
    ],
    entry_points={
        'console_scripts': [
            'inspire=inspire.run:main'
        ]
    },
    include_package_data=True,
    package_data={'': ['model/*.pkl']},
    install_requires=[
        'biopython==1.79',
        'certifi==2021.10.8',
        'cycler==0.11.0',
        'fonttools==4.30.0',
        'joblib==1.1.0',
        'kiwisolver==1.3.2',
        'llvmlite==0.38.0',
        'lxml==4.8.0',
        'matplotlib==3.5.1',
        'mokapot==0.8.0',
        'numba==0.55.1',
        'numpy==1.21.5',
        'packaging==21.3',
        'pandas==1.4.1',
        'Pillow==9.0.1',
        'plotly==5.6.0',
        'pyopenms==2.7.0',
        'pyparsing==3.0.7',
        'pyteomics==4.5.3',
        'python-dateutil==2.8.2',
        'pytz==2021.3',
        'PyYAML==6.0',
        'scikit-learn==1.0.2',
        'scipy==1.8.0',
        'six==1.16.0',
        'tenacity==8.0.1',
        'threadpoolctl==3.1.0',
        'triqler==0.6.2',
    ],
)
