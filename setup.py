"""
CogPrime: An Integrated AGI Architecture
Setup configuration for the CogPrime package and its integrated components.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() 
        for line in fh 
        if line.strip() and not line.startswith("#")
    ]

# Read dev requirements
with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
    dev_requirements = [
        line.strip() 
        for line in fh 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="cogprime",
    version="0.1.0",
    author="CogPy Team",
    author_email="d@rzo.io",
    description="An Integrated AGI Architecture combining OpenCog, Hyperon, and Vervaeke's Relevance Realization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cogpy/cogprime",
    packages=find_packages(where="src") + find_packages(where="integrations"),
    package_dir={
        "": "src",
        "mem0": "integrations/mem0/mem0",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "graph": ["langchain-community", "neo4j", "rank-bm25"],
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "cogprime=core.silicon_sage_core:main",
        ],
    },
)
