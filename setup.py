import pathlib
from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    name="geohexviz",
    version="1.0.0",
    description="A library for the visualization of hexagonally binned data sets.",
    long_description=readme,
    long_description_content_type="text/x-rst",
    url="",
    author="Tony Marco Abou Zeidan",
    author_email="tony.azp25@gmail.com",
    license="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9.4",
        "License :: OSI Approved :: BSD License",   # TODO: ensure correct licensing
        "Operating System :: OS Independent",  # TODO: ensure deployment on other OS
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords='visualization,geospatial,hexagonal binning',
    python_requires=">=3.6",
    packages=find_packages(exclude=["docs", "tests"]),
    include_package_data=True,
    install_requires=[
        "h3~=3.7.0",
        "shapely~=1.7.1",
        "pyproj~=3.0.0.post1",
        "numpy~=1.20.3",
        "geojson~=2.5.0",
        "pandas~=1.3.0",
        "plotly~=4.14.3",
        "kaleido~=0.2.1",
        "pyyaml~=5.4.1",
        "rtree~=0.9.1",
        "openpyxl~=3.0.9"
    ],
    extras_require={
        "dev": [
            "pip",
            "Sphinx",
            "openpyxl~=3.0.9",
            "pdfcropmargins~=1.0.5"
        ],
        "pdf-crop": [
            "pdfcropmargins~=1.0.5"
        ]
    },
    entry_points={
        "console_scripts": ["geohexsimple=geohexsimple.simple:main"]
    }
)

print(find_packages(exclude=tuple("tests")))
