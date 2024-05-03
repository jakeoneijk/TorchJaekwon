# pip install -e ./

from setuptools import setup, find_packages

setup(
    name='TorchJaekwon',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here.
        # They will be installed by pip when your project is installed.
        'numpy',
        'tqdm',
        'psutil',
        'pyyaml',
        'matplotlib',
        'librosa',
        'wandb',
        'tensorboardX',
    ],
    # additional metadata about your project
    author='Jaekwon Im',
    author_email='jakeoneijk@kaist.ac.kr',
    description='',
    license='',
    keywords='',
)