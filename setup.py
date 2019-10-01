import setuptools
from niceback import __version__

setuptools.setup(
    name="niceback",
    version=__version__,
    author="L. Kärkkäinen",
    author_email="tronic@noreply.users.github.com",
    description="Human-readable tracebacks in Jupyter notebooks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Tronic/niceback",
    packages=setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "License :: Public Domain",
        "Operating System :: OS Independent",
    ],
    install_requires = [
    ],
    include_package_data = True,
)
