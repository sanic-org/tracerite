from setuptools import setup, find_packages

setup(
    name="niceback",
    author="L. Kärkkäinen",
    author_email="tronic@noreply.users.github.com",
    description="Human-readable tracebacks in Jupyter notebooks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Tronic/niceback",
    use_scm_version=True,
    setup_requires = ["setuptools_scm"],
    packages=find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "License :: Public Domain",
        "Operating System :: OS Independent",
    ],
    install_requires = ["html5tagger>=1.0.0"],
    include_package_data = True,
)
