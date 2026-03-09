"""
Minimal setup – makes the project pip-installable inside the venv.
"""
from setuptools import setup, find_packages

setup(
    name="chronic_chatbot",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
