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
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",   
        "Operating System :: OS Independent",  # TODO: ensure deployment on other OS
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering"
    ],
    keywords='visualization,geospatial,hexagonal binning',
    python_requires=[">=3.7", "<=3.8.16"],
    packages=find_packages(exclude=["docs", "tests"]),
    include_package_data=True,
    install_requires=[
        "h3==3.7.0",
        "shapely==^1.7.1",
        "geojson==2.5.0",
        "plotly==4.14.3",
        "kaleido==0.2.1",
        "pyyaml==5.4.1",
        "rtree==0.9.7",
        "openpyxl==3.0.9"   # required for XLSX input
    ],
    extras_require={
        "dev": [
            "pip",
            "Sphinx",
            "pdfcropmargins==1.0.5"
        ],
        "pdf-crop": [
            "pdfcropmargins~=1.0.5"
        ]
    },
    entry_points={
        'console_scripts': ['geohexsimple=geohexviz.simple:main'],
    }
)

print(find_packages(exclude=tuple("tests")))
