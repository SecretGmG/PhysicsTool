from setuptools import setup, find_packages


setup(
    name="PhysicsTool",
    version="2.5",
    packages=find_packages(),
    requires=['numpy', 'sympy']
)