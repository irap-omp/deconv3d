from distutils.core import setup
import os

# Build and publish :
# python setup.py sdist upload

version = '1.0.0'
github_url = 'https://github.com/irap-omp/deconv3d'
author = 'Antoine Goutenoir'
email = 'antoine.goutenoir@irap.omp.eu'


# Pypi does not support markdown (the cheeseshop strikes again),
# so we're converting our MarkDown README into ReStructuredText.
long_description = ''
try:
    import pypandoc  # to convert from markdown to restructured text
    long_description = pypandoc.convert('README.md', 'rst')
    long_description = long_description.replace("\r", "")
except (ImportError, OSError):
    print("Pandoc not found. Long description conversion failure.")
    import io
    with io.open('README.md', encoding="utf-8") as f:
        long_description = f.read()


setup(
    name='deconv3d',
    py_modules=['lib/run'],
    version=version,
    description='An implementation of a MCMC with MH within Gibbs for the '
                'estimation and non-parametric deconvolution of galactic '
                'kinematics from hyperspectral data cubes.',
    author=author,
    author_email=email,
    maintainer=author,
    maintainer_email=email,
    url=github_url,
    download_url='{}/tarball/{}'.format(github_url, version),
    keywords=['deconvolution', 'science', 'gibbs', 'markov', 'runner'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Topic :: Utilities",
    ],
    long_description=long_description,
    license='MIT'
)
