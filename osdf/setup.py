from setuptools import setup, find_packages

setup(
    name="odsf",
    version="0.1.0",
    author="David Wynter",
    description="Objective-Driven Stochastic Fields Framework",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "networkx>=3.0",
        "torch>=2.0",
        "pandas>=2.0",
        "matplotlib>=3.7"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "flake8>=6.0",
            "sphinx>=6.0"
        ],
        "viz": ["matplotlib>=3.7"]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.9',
)