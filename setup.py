import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

PACKAGE_DATA = {
    'miner': ['data/*', 'data/network_dictionaries/*']
    }

INSTALL_REQUIRES = ['numpy', 'scipy', 'pandas', 'sklearn', 'lifelines',
                    'matplotlib', 'seaborn', 'mygene',
                    'pydot', 'graphviz', 'opentargets']
setuptools.setup(
    name="isb_miner3",
    version="1.0.7",
    author="Matt Wall",
    author_email="mwall@systemsbiology.org",
    description="MINER analysis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baliga-lab/miner3",
    packages=['miner'],
    install_requires = INSTALL_REQUIRES,
    include_package_data=True, package_data=PACKAGE_DATA,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    scripts=['bin/miner3-coexpr', 'bin/miner3-mechinf',
             'bin/miner3-bcmembers', 'bin/miner3-subtypes',
             'bin/miner3-survival', 'bin/miner3-causalinference', 'bin/miner3-causalinf-pre',
             'bin/miner3-causalinf-post', 'bin/miner3-neo', 'bin/miner3-riskpredict',
             'bin/gene2opentargets', 'bin/drug2opentargets'])
