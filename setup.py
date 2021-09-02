import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="geohexviz",  # formerly real-geohexviz
    version="1.0.0",
    description="A library for the visualization of hexagonally binned data sets.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="",
    author="Tony Abou Zeidan",
    author_email="tony.azp25@gmail.com",
    license="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9.4"
    ],
    packages=find_packages(exclude=tuple("tests")),
    include_package_data=True,
    install_requires=["h3", "shapely", "pyproj", "numpy", "pandas", "gdal", "fiona", "geopandas", "plotly", "kaleido"]
)
