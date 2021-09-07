import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()
print("HERE:", HERE)

setup(
    name="geohexviz",  # formerly real-geohexviz
    version="1.0.0",
    description="A library for the visualization of hexagonally binned data sets.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="",
    author="Tony Marco Abou Zeidan",
    author_email="tony.azp25@gmail.com",
    license="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9.4"
    ],
    python_requires=">=3.1",
    packages=find_packages(exclude=tuple("tests")),
    include_package_data=True,
    install_requires=["h3", "shapely", "pyproj", "numpy", "geojson", "pandas", "plotly", "kaleido"]
)

print('PACKAGES:', find_packages(exclude=tuple("tests")))
