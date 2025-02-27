from setuptools import setup, find_packages

setup(
    name='ayaka_utils',
    version='0.0.0',
    author='Lexa-B',
    author_email='Lexa.40@proton.me',
    description='A collection of utilities to be shared across the AyakaAI codebase',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Lexa-B/AyakaUtils',
    packages=find_packages(),       # automatically find packages (will pick up ayaka_utils)
    install_requires=[],            # list runtime dependencies if any
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
