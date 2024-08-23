from setuptools import setup, find_packages

setup(
    name="score-life-programming",
    version="0.1.0",
    author="Abhinav Muraleedharan",
    author_email="Abhinav.Muraleedharan@mail.utoronto.ca",
    description="Implementation of Score-life programming for reinforcement learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/score-life-programming",  # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "scipy",
        "numpy",
        "seaborn",
        "matplotlib",
        "scikit-learn",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)