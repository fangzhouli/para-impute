from setuptools import setup, find_packages

with open('README.md', 'r') as file:
    long_description = file.read()

setup(
    name                = 'para-impute',
    version             = '1.0.0',
    packages            = find_packages(exclude=["tests"]),
    license             = 'MIT',
    description         = 'Missing value imputation package for high-performance computing',
    author              = 'Fangzhou Li',
    author_email        = 'fzli0805@gmail.com',
    url                 = 'https://github.com/fangzhouli/para-impute',
    # download_url = 'https://github.com/fangzhouli/para-impute/archive/v_01.tar.gz',
    keywords            = [
        'impute',
        'imputation',
        'missing data',
        'missing value',
        'missing value imputation',
        'random forest',
        'HPC',
        'high-performance computing',
        'computer cluster',
        'SLURM'],
    install_requires    = [
            'numpy',
            'scikit-learn'],
    long_description    = long_description,
    long_description_content_type = 'text/markdown',
    classifiers         = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3']
)