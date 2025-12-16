from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sam',
    version="0.0.1",
    author='Anonymous',
    author_email='anonymous@anonymous.org',
    description='Sparse Abstract Machine',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    python_requires=">=3.5",
    packages=[
        "sam",
        "sam.sim",
    ],
    install_requires=[]
)
