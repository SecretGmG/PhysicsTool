from setuptools import setup, find_packages


setup(
    name="PhysicsTool",
    version="3.9",
    packages=find_packages(),
    requires=['numpy', 'sympy']
)