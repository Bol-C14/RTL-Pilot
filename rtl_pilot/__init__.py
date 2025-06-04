"""
RTL-Pilot: Multi-agent LLM application for automating RTL verification workflows.

This package provides automated testbench generation, simulation running,
and result evaluation for RTL designs using LLM agents.
"""

__version__ = "0.1.0"
__author__ = "RTL-Pilot Team"

from .config.settings import Settings
from .workflows.default_flow import DefaultVerificationFlow

__all__ = [
    "Settings",
    "DefaultVerificationFlow",
]
