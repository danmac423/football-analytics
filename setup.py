from setuptools import setup, find_packages

setup(
    name="football-analytics",
    version="0.1.0",
    packages=find_packages(where="football_analytics"),
    package_dir={"": "football_analytics"},
)
