from setuptools import setup, find_packages

setup(
    name="tracerite",
    author="Sanic Community",
    author_email="tronic@noreply.users.github.com",
    description="Human-readable HTML tracebacks for Python exceptions",
    long_description=open("README.md", encoding="UTF-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sanic-org/tracerite",
    use_scm_version=True,
    setup_requires = ["setuptools_scm"],
    packages=find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "License :: Public Domain",
        "Operating System :: OS Independent",
    ],
    install_requires = ["html5tagger>=1.2.1"],
    include_package_data = True,
)
