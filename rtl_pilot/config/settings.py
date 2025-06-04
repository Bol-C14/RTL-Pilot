"""
Global settings and configuration for RTL-Pilot.

This module manages all configuration settings including tool paths,
LLM configurations, and verification parameters.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging


@dataclass
class Settings:
    """
    Global configuration settings for RTL-Pilot.
    """
    
    # Tool paths
    vivado_path: Path = field(default_factory=lambda: Path("/opt/Xilinx/Vivado/2024.1/bin/vivado"))
    modelsim_path: Optional[Path] = None
    questa_path: Optional[Path] = None
    
    # LLM configuration with LangChain support
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None
    llm_api_base: Optional[str] = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4000
    llm_enable_tool_calling: bool = True
    llm_max_tool_calls: int = 10
    llm_tool_timeout: float = 30.0
    llm_enable_streaming: bool = False
    llm_langsmith_tracing: bool = False
    
    # Directories
    prompts_dir: Path = field(default_factory=lambda: Path("./rtl_pilot/prompts"))
    output_dir: Path = field(default_factory=lambda: Path("./rtl_pilot_output"))
    temp_dir: Path = field(default_factory=lambda: Path("/tmp/rtl_pilot"))
    
    # Simulation settings
    default_sim_time: str = "1us"
    simulation_timeout: str = "10ms"
    enable_waveforms: bool = True
    enable_coverage: bool = True
    
    # Verification settings
    default_coverage_target: float = 90.0
    max_verification_iterations: int = 5
    parallel_simulations: int = 1
    
    # Agent settings
    testbench_template: str = "verilog_tb.jinja2"
    feedback_template: str = "feedback_loop.jinja2"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    # Advanced settings
    debug_mode: bool = False
    save_intermediate_files: bool = True
    auto_cleanup: bool = False
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure paths are Path objects
        self.vivado_path = Path(self.vivado_path)
        self.prompts_dir = Path(self.prompts_dir)
        self.output_dir = Path(self.output_dir)
        self.temp_dir = Path(self.temp_dir)
        
        if self.modelsim_path:
            self.modelsim_path = Path(self.modelsim_path)
        if self.questa_path:
            self.questa_path = Path(self.questa_path)
        if self.log_file:
            self.log_file = Path(self.log_file)
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load API key from environment if not set
        if not self.llm_api_key:
            self.llm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    
    def load_from_file(self, config_file: Path) -> None:
        """
        Load settings from a JSON configuration file.
        
        Args:
            config_file: Path to configuration file
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        self.load_from_dict(config_data)
    
    def load_from_dict(self, config_data: Dict[str, Any]) -> None:
        """
        Load settings from a dictionary.
        
        Args:
            config_data: Dictionary containing configuration values
        """
        for key, value in config_data.items():
            if hasattr(self, key):
                # Convert path strings to Path objects
                if key.endswith("_path") or key.endswith("_dir") or key == "log_file":
                    if value is not None:
                        value = Path(value)
                
                setattr(self, key, value)
    
    def save_to_file(self, config_file: Path) -> None:
        """
        Save current settings to a JSON configuration file.
        
        Args:
            config_file: Path to save configuration file
        """
        config_data = self.to_dict()
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.
        
        Returns:
            Dictionary representation of settings
        """
        config_dict = {}
        
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        return config_dict
    
    def validate(self) -> List[str]:
        """
        Validate configuration settings.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check tool paths
        if not self.vivado_path.exists():
            errors.append(f"Vivado path does not exist: {self.vivado_path}")
        
        if self.modelsim_path and not self.modelsim_path.exists():
            errors.append(f"ModelSim path does not exist: {self.modelsim_path}")
        
        if self.questa_path and not self.questa_path.exists():
            errors.append(f"QuestaSim path does not exist: {self.questa_path}")
        
        # Check prompts directory
        if not self.prompts_dir.exists():
            errors.append(f"Prompts directory does not exist: {self.prompts_dir}")
        
        # Check LLM configuration
        if not self.llm_api_key:
            errors.append("LLM API key not configured")
        
        # Check simulation settings
        if self.default_coverage_target < 0 or self.default_coverage_target > 100:
            errors.append(f"Invalid coverage target: {self.default_coverage_target}")
        
        if self.max_verification_iterations < 1:
            errors.append(f"Invalid max iterations: {self.max_verification_iterations}")
        
        return errors
    
    def get_vivado_command(self) -> List[str]:
        """
        Get Vivado command with proper arguments.
        
        Returns:
            List of command arguments for Vivado
        """
        return [str(self.vivado_path), "-mode", "batch", "-nojournal", "-nolog"]
    
    def get_simulator_command(self, simulator: str = "vivado") -> List[str]:
        """
        Get simulator command based on configured simulator.
        
        Args:
            simulator: Simulator to use ("vivado", "modelsim", "questa")
            
        Returns:
            List of command arguments for the simulator
        """
        if simulator == "vivado":
            return self.get_vivado_command()
        elif simulator == "modelsim" and self.modelsim_path:
            return [str(self.modelsim_path)]
        elif simulator == "questa" and self.questa_path:
            return [str(self.questa_path)]
        else:
            raise ValueError(f"Simulator '{simulator}' not configured or not supported")
    
    def setup_logging(self) -> None:
        """Setup logging configuration based on settings."""
        handlers = [logging.StreamHandler()]
        
        if self.log_file:
            handlers.append(logging.FileHandler(self.log_file))
        
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers
        )
    
    @classmethod
    def load_default(cls) -> "Settings":
        """
        Load default settings, checking for config file in standard locations.
        
        Returns:
            Settings instance with default or loaded configuration
        """
        settings = cls()
        
        # Check for config files in standard locations
        config_paths = [
            Path("rtl_pilot_config.json"),
            Path("~/.rtl_pilot/config.json").expanduser(),
            Path("/etc/rtl_pilot/config.json")
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    settings.load_from_file(config_path)
                    logging.info(f"Loaded configuration from: {config_path}")
                    break
                except Exception as e:
                    logging.warning(f"Failed to load config from {config_path}: {e}")
        
        return settings

# Global settings instance
settings = Settings()

# Alias for backwards compatibility and cleaner imports
RTLPilotSettings = Settings

def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings: Global configuration settings
    """
    return settings

def update_settings(**kwargs) -> None:
    """
    Update global settings with new values.
    
    Args:
        **kwargs: Settings to update
    """
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            raise ValueError(f"Unknown setting: {key}")

def load_settings_from_file(config_file: Path) -> None:
    """
    Load settings from configuration file.
    
    Args:
        config_file: Path to configuration file
    """
    global settings
    settings.load_from_file(config_file)

def reset_settings() -> None:
    """Reset settings to default values."""
    global settings
    settings = Settings()
