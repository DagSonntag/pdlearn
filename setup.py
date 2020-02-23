from setuptools import setup, find_packages

setup(
    name="pdlearn",
    version="0.0.1",
    description="A pandas wrapper library for sklearn",
    packages=find_packages(),
    author='Dag Sonntag',
    author_email='dag@sonntag.se',
    url="add", install_requires=['scikit-learn', 'pandas']
)