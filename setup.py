import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

PACKAGE_DATA = {
    'miner': ['data/*', 'data/network_dictionaries/*']
    }

INSTALL_REQUIRES = ['numpy>=1.22.0', 'scipy>=1.8.0', 'pandas>=2.0.0', 'scikit-learn>=1.1.3',
                    'lifelines>=0.27.4',
                    'matplotlib>=3.6.0',
                    'seaborn>=0.12.0',
                    'mygene>=3.2.2',
                    'requests_toolbelt>=1.0.0',
                    'pydot>=2.0.0',
                    'graphviz>=0.20.1',
                    'gql>=3.4.0',
                    'chembl_webresource_client>=0.10.8',
                    'tqdm>=4.64.0']
setuptools.setup(
    name="isb_miner3",
    version="1.2.3",
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
             'bin/miner3-subtypes',
             'bin/miner3-survival',
             'bin/miner3-causalinference',
             'bin/miner3-riskpredict',
             'bin/gene2opentargets', 'bin/drug2opentargets'])
