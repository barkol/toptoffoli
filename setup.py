from setuptools import setup, find_packages

setup(
    name="toffoli-optimizer",
    version="1.0.0",
    author="Karol Bartkiewicz",
    author_email="karol.bartkiewicz@example.com",
    description="A tool for optimizing Toffoli networks for quantum circuits",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bartkiewicz/toffoli-optimizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Quantum Computing",
    ],
    python_requires=">=3.6",
    install_requires=[
        "qiskit>=2.0.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
        ],
    },
)
