import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kekas",
    version="0.1.17",
    author="Aleksandr Belskikh",
    author_email="belskikh.aleksandr@gmail.com",
    description="Just another DL library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/belskikh/kekas",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires=">=3.6.0",
    packages=setuptools.find_packages(),
    install_requires=[
        "pandas>=0.22",
        "numpy>=1.14.6",
        "plotly>=3.6.1",
        "tensorboard==1.12.2",
        "tensorboardX>=1.6",
        "tensorflow==1.12",
        "torch>=0.4.1",
        "torchvision>=0.2.1",
        "tqdm>=4.29.1",
        "scikit-learn>=0.20"
    ]
)
