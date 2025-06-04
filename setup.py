#!/usr/bin/env python3
"""
RTL-Pilot Setup Script

This script handles the installation and setup of RTL-Pilot,
a multi-agent LLM framework for RTL verification automation.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read version from package
def get_version():
    """Get version from package __init__.py"""
    init_file = Path(__file__).parent / "rtl_pilot" / "__init__.py"
    if init_file.exists():
        with open(init_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return "0.1.0"

# Read long description from README
def get_long_description():
    """Get long description from README file"""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def get_requirements():
    """Get requirements from requirements.txt"""
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        with open(req_file, 'r') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove version comments and extra whitespace
                    requirement = line.split('#')[0].strip()
                    if requirement:
                        requirements.append(requirement)
            return requirements
    return []

# Optional dependencies
def get_optional_requirements():
    """Get optional dependency groups"""
    return {
        'dev': [
            'black>=23.0.0',
            'isort>=5.12.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
            'pre-commit>=3.0.0',
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'pytest-mock>=3.11.0',
            'pytest-cov>=4.1.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.3.0',
            'myst-parser>=2.0.0',
        ],
        'web': [
            'fastapi>=0.104.0',
            'uvicorn>=0.24.0',
            'websockets>=11.0.0',
        ],
        'visualization': [
            'matplotlib>=3.7.0',
            'plotly>=5.17.0',
        ],
        'hardware': [
            'pyserial>=3.5',
            'paramiko>=3.3.0',
        ]
    }

setup(
    name="rtl-pilot",
    version=get_version(),
    author="RTL-Pilot Development Team",
    author_email="rtl-pilot@example.com",
    description="Multi-Agent LLM Framework for RTL Verification Automation",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/rtl-pilot/rtl-pilot",
    project_urls={
        "Bug Tracker": "https://github.com/rtl-pilot/rtl-pilot/issues",
        "Documentation": "https://rtl-pilot.readthedocs.io/",
        "Source Code": "https://github.com/rtl-pilot/rtl-pilot",
    },
    
    # Package information
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    package_data={
        'rtl_pilot': [
            'prompts/*.jinja2',
            'scripts/*.tcl',
            'config/*.yaml',
            'config/*.json',
        ],
    },
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require=get_optional_requirements(),
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'rtl-pilot=rtl_pilot.interface.cli:main',
            'rtl-pilot-web=rtl_pilot.interface.web_ui:main',
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Code Generators",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: Web Environment",
    ],
    
    # Keywords for search
    keywords=[
        "rtl", "verification", "testbench", "verilog", "systemverilog", 
        "vivado", "llm", "ai", "automation", "eda", "fpga", "asic",
        "multi-agent", "code-generation", "testing"
    ],
    
    # Additional metadata
    license="MIT",
    zip_safe=False,
    
    # Test configuration
    test_suite="tests",
    tests_require=[
        'pytest>=7.0.0',
        'pytest-asyncio>=0.21.0',
        'pytest-mock>=3.11.0',
    ],
    
    # Options for different build scenarios
    options={
        'bdist_wheel': {
            'universal': False,  # Not universal since we support Python 3.8+
        },
        'egg_info': {
            'tag_build': '',
            'tag_date': False,
        },
    },
)
