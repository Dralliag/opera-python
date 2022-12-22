from setuptools import setup
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.1"
DESCRIPTION = (
    "Python version of opera package https://cran.r-project.org/web/packages/opera/"
)

setup(
    name="opera",
    version=VERSION,
    author="Riad Ladjel",
    author_email="<riad.ladjel@quadratic-labs.com>",
    description=DESCRIPTION,
    url="https://github.com/Dralliag/opera-python",
    long_description_content_type="text/markdown",
    long_description=long_description,
    license="MIT",
    packages=["opera"],
    install_requires=["pandas", "numpy", "matplotlib", "seaborn"],
    setup_requires=["wheel"],
    tests_require=["pytest"],
)
