from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ensemble-llm",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-model LLM ensemble with voting and web search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ensemble-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.9.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "colorama>=0.4.6",
    ],
    entry_points={
        "console_scripts": [
            "ensemble-llm=ensemble_llm.main:main",
        ],
    },
)