from setuptools import setup, find_packages

setup(
    name="ML",
    version="0.1",
    packages=find_packages(),
    scripts=['train_model.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['tensorflow==1.10', 'nltk', 'numpy', 'pandas', 'gensim'],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt'],
    },

    # metadata to display on PyPI
    author="Sameer Hameed",
    author_email="samorsameer@gmail.com",
    keywords="Abstract summarization",
    url="https://github.com/simplysameer333/MachineLearning/tree/master/POC",  # project home page, if any
    project_urls={
        "Source Code": "https://github.com/simplysameer333/MachineLearning/tree/master/POC",
    }

    # could also include long_description, download_url, classifiers, etc.
)

# python setup.py install
