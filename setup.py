from setuptools import find_packages, setup

setup(
    name="micro-q",
    version="0.1.0",
    description="Micro-Q: Autonomous Materials Discovery Pipeline — "
    "a proof-of-concept mimicking Novyte's core product Q.",
    packages=find_packages(),
    python_requires=">=3.11",
)
