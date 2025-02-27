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
    install_requires=[
        "langchain==0.3.17", # For LangChain
        "langchain_nvidia_ai_endpoints==0.3.9", # For NVIDIA AI Endpoints
        "langchain_openai==0.3.4", # For OpenAI
        "dotenv==0.9.9", # For loading environment variables
        "hjson==3.1.0", # Even though this is for hjson, we use it for JSONC files, such as the model config
        "rich==13.9.4", # For pretty printing
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
