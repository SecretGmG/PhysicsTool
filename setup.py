from setuptools import setup, find_packages # type: ignore


setup(
    name="PhysicsTool",
    version="5.4",
    packages=find_packages(),
    requires=["numpy", "sympy", "pandas", "matplotlib", "scipy"],
)

