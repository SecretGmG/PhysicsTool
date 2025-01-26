from setuptools import setup, find_packages


setup(
    name="PhysicsTool",
    version="4.3",
    packages=find_packages(),
    requires=['numpy', 'sympy']
)