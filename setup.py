import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kekas",
    version="0.1",
    author="Aleksandr Belskikh",
    author_email="belskikh.aleksandr@gmail.com",
    description="A simple library to train neural networks with pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/belskikh/kekas",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
