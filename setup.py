from setuptools import setup

setup(
    name='superdea',
    version='0.1',
    description='Superefficiency of DEA',
    url='',
    author='Miriam Esteve and Juan Aparicio',
    author_email='miriam.estevec@umh.es',
    packages=['superdea'],
    install_requires=['numpy', 'pandas', 'graphviz', 'docplex', "matplotlib", "scipy"],
    license='AFL-3.0',
    zip_safe=False
)