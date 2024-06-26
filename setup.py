from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Hi-C analysis tools"
LONG_DESCRIPTION = "Tools for downstream analysis of Hi-C data such as creating scaling plots or calculating insulation scores."

setup(
    name="hiC",
    version=VERSION,
    author="Emma Rusch",
    author_email="rusch_emma@icloud.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
)
    