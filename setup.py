from setuptools import setup, find_packages # type: ignore


setup(
    name="PhysicsTool",
    version="5.0",
    packages=find_packages(),
    requires=["numpy", "sympy", "pandas", "matplotlib", "scipy"],
)

