from setuptools import setup

setup(
    name='visanalysis',
    version='0.1.0',
    description='Analysis environment for visprotocol experiments',
    url='https://github.com/ClandininLab/visanalysis',
    author='Max Turner',
    author_email='mhturner@stanford.edu',
    packages=['visanalysis'],
    install_requires=['PyQT5',
        'numpy',
        'h5py',
        'scipy',
        'pandas',
        'scikit-image',
        'thunder-registration',
        'seaborn',
        'pyyaml',
        'matplotlib',
        'npTDMS',
        'nibabel'],
    include_package_data=True,
    zip_safe=False,
)
