import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MLPemselble",
    version="0.1.0",
    author="Masaaki Kotera",
    author_email="maskot1977@gmail.com",
    description="Multi-Layer Perceptron Ensemble",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maskot1977/MLPensemble/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": ["sample_command = sample_command.sample_command:main"]
    },
    python_requires=">=3.6",
)
