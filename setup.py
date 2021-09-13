import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()
VERSION = "1.0.0"
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
    install_requires=["h3>=3.7.0", "shapely>=1.7.1", "pyproj", "numpy", "geojson>=2.5.0", "pandas", "plotly", "kaleido"],
    entry_points="""
    [console_scripts]
    geosimple = scripts.geosimple.simple:main
    """,
)

print('PACKAGES:', find_packages(exclude=tuple("tests")))
print(f"GeoHexViz v{VERSION} has been installed.")
