from setuptools import setup, find_packages


setup(
    name="PhysicsTool",
    version="3.4",
    packages=find_packages(),
    requires=['numpy', 'sympy']
)