"""
CogPrime: An Integrated AGI Architecture
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cogprime",
    version="1.0.0",
    author="CogPrime Development Team",
    author_email="dev@cogprime.org",
    description="An Integrated AGI Architecture combining OpenCog Prime, Hyperon, and Vervaeke's Relevance Realization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cogpy/cogprime",
    project_urls={
        "Bug Tracker": "https://github.com/cogpy/cogprime/issues",
        "Documentation": "https://github.com/cogpy/cogprime/tree/main/docs",
        "Source Code": "https://github.com/cogpy/cogprime",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "ruff>=0.0.270",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cogprime=core.cognitive_core:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
