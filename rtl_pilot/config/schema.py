"""
Configuration schemas and validation for RTL-Pilot.

This module defines Pydantic schemas for configuration validation
and data serialization/deserialization.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
import logging
import time


class ToolPaths(BaseModel):
    """Configuration for tool paths."""
    
    vivado_path: Path = Field(
        default=Path("/opt/Xilinx/Vivado/2024.1/bin/vivado"),
        description="Path to Vivado executable"
    )
    modelsim_path: Optional[Path] = Field(
        default=None,
        description="Path to ModelSim executable"
    )
    questa_path: Optional[Path] = Field(
        default=None,
        description="Path to QuestaSim executable"
    )
    
    @field_validator("vivado_path", "modelsim_path", "questa_path")
    @classmethod
    def validate_tool_path(cls, v):
        """Validate that tool paths exist if specified."""
        if v is not None and not v.exists():
            logging.warning(f"Tool path does not exist: {v}")
        return v


class LLMConfig(BaseModel):
    """Configuration for LLM settings with LangChain support."""
    
    provider: str = Field(
        default="openai",
        description="LLM provider: 'openai', 'anthropic', 'local'"
    )
    model: str = Field(
        default="gpt-4",
        description="LLM model to use"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for LLM service"
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for LLM API"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM"
    )
    max_tokens: int = Field(
        default=4000,
        gt=0,
        description="Maximum tokens for LLM response"
    )
    
    # LangChain specific settings
    enable_tool_calling: bool = Field(
        default=True,
        description="Enable LLM tool calling capabilities"
    )
    max_tool_calls: int = Field(
        default=10,
        description="Maximum number of tool calls per interaction"
    )
    tool_timeout: float = Field(
        default=30.0,
        description="Timeout for tool execution in seconds"
    )
    enable_streaming: bool = Field(
        default=False,
        description="Enable streaming responses"
    )
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing for debugging"
    )
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v):
        """Validate LLM provider."""
        supported_providers = ["openai", "anthropic", "local"]
        if v not in supported_providers:
            logging.warning(f"Unsupported provider '{v}', supported: {supported_providers}")
        return v
    
    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        """Validate LLM model name."""
        supported_models = [
            "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o",
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3-5-sonnet",
            "llama2", "llama3", "codellama", "local-model"
        ]
        if v not in supported_models:
            logging.warning(f"Unsupported model '{v}', supported: {supported_models}")
        return v


class DirectoryConfig(BaseModel):
    """Configuration for directory paths."""
    
    prompts_dir: Path = Field(
        default=Path("./rtl_pilot/prompts"),
        description="Directory containing prompt templates"
    )
    output_dir: Path = Field(
        default=Path("./rtl_pilot_output"),
        description="Default output directory"
    )
    temp_dir: Path = Field(
        default=Path("/tmp/rtl_pilot"),
        description="Temporary files directory"
    )
    
    @field_validator("prompts_dir", "output_dir", "temp_dir")
    @classmethod
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class SimulationConfig(BaseModel):
    """Configuration for simulation settings."""
    
    default_sim_time: str = Field(
        default="1us",
        description="Default simulation time"
    )
    simulation_timeout: str = Field(
        default="10ms",
        description="Simulation timeout"
    )
    enable_waveforms: bool = Field(
        default=True,
        description="Enable waveform generation"
    )
    enable_coverage: bool = Field(
        default=True,
        description="Enable coverage analysis"
    )
    parallel_simulations: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of parallel simulations"
    )
    
    @field_validator("default_sim_time", "simulation_timeout")
    @classmethod
    def validate_time_format(cls, v):
        """Validate time format (e.g., 1us, 10ns, 1ms)."""
        import re
        pattern = r'^\d+(\.\d+)?(ns|us|ms|s)$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid time format: {v}. Use format like '1us', '10ns', '1ms'")
        return v


class VerificationConfig(BaseModel):
    """Configuration for verification settings."""
    
    default_coverage_target: float = Field(
        default=90.0,
        ge=0.0,
        le=100.0,
        description="Default coverage target percentage"
    )
    max_verification_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum verification iterations"
    )
    testbench_template: str = Field(
        default="verilog_tb.jinja2",
        description="Default testbench template"
    )
    feedback_template: str = Field(
        default="feedback_loop.jinja2",
        description="Default feedback template"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Log file path"
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Valid levels: {valid_levels}")
        return v.upper()


class AdvancedConfig(BaseModel):
    """Configuration for advanced settings."""
    
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    save_intermediate_files: bool = Field(
        default=True,
        description="Save intermediate files for debugging"
    )
    auto_cleanup: bool = Field(
        default=False,
        description="Automatically cleanup temporary files"
    )


class VerificationScenario(BaseModel):
    """Schema for test scenario definition."""
    
    name: str = Field(description="Test scenario name")
    description: str = Field(description="Test scenario description")
    priority: str = Field(
        default="medium",
        description="Test priority (low, medium, high)"
    )
    test_vectors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of test vectors"
    )
    expected_coverage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Expected coverage for this scenario"
    )
    
    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        """Validate test priority."""
        valid_priorities = ["low", "medium", "high", "critical"]
        if v.lower() not in valid_priorities:
            raise ValueError(f"Invalid priority: {v}. Valid priorities: {valid_priorities}")
        return v.lower()


class VerificationGoals(BaseModel):
    """Schema for verification goals."""
    
    coverage_target: float = Field(
        default=90.0,
        ge=0.0,
        le=100.0,
        description="Target coverage percentage"
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum verification iterations"
    )
    performance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Performance requirements"
    )
    functional_requirements: Optional[List[str]] = Field(
        default=None,
        description="Functional requirements to verify"
    )
    timeout: Optional[str] = Field(
        default=None,
        description="Overall verification timeout"
    )


class RTLDesignInfo(BaseModel):
    """Schema for RTL design information."""
    
    module_name: str = Field(description="Top-level module name")
    language: str = Field(
        default="verilog",
        description="RTL language (verilog, systemverilog)"
    )
    ports: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Module ports information"
    )
    parameters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Module parameters"
    )
    clock_domains: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Clock domain information"
    )
    reset_signals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Reset signals information"
    )
    complexity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Design complexity score"
    )
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        """Validate RTL language."""
        valid_languages = ["verilog", "systemverilog", "vhdl"]
        if v.lower() not in valid_languages:
            raise ValueError(f"Invalid language: {v}. Valid languages: {valid_languages}")
        return v.lower()


class SimulationResults(BaseModel):
    """Schema for simulation results."""
    
    success: bool = Field(description="Simulation success status")
    runtime: float = Field(ge=0.0, description="Simulation runtime in seconds")
    errors: List[str] = Field(default_factory=list, description="Simulation errors")
    warnings: List[str] = Field(default_factory=list, description="Simulation warnings")
    coverage: Optional[Dict[str, float]] = Field(
        default=None,
        description="Coverage metrics"
    )
    waveform_file: Optional[Path] = Field(
        default=None,
        description="Path to waveform file"
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Path to simulation log file"
    )


class EvaluationResults(BaseModel):
    """Schema for evaluation results."""
    
    overall_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall evaluation score"
    )
    pass_status: bool = Field(description="Overall pass/fail status")
    functional_correctness: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Functional correctness analysis"
    )
    coverage_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Coverage analysis results"
    )
    performance_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Performance metrics"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Improvement recommendations"
    )
    issues_found: List[str] = Field(
        default_factory=list,
        description="Issues identified"
    )


class ConfigSchema(BaseModel):
    """Complete configuration schema for RTL-Pilot."""
    
    tool_paths: ToolPaths = Field(default_factory=ToolPaths)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    directories: DirectoryConfig = Field(default_factory=DirectoryConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    
    model_config = {"validate_assignment": True, "extra": "forbid"}
        
    @model_validator(mode='before')
    @classmethod
    def validate_config(cls, values):
        """Perform cross-field validation."""
        # Ensure API key is set for non-local models
        llm_config = values.get("llm")
        if llm_config and llm_config.model != "local-model" and not llm_config.api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logging.warning("No API key found for LLM service")
        
        return values
    
    def to_settings(self):
        """Convert to Settings object."""
        from .settings import Settings
        
        # Create settings object with flattened configuration
        settings_dict = {}
        
        # Flatten nested configuration
        for section_name, section_config in self.dict().items():
            if isinstance(section_config, dict):
                for key, value in section_config.items():
                    settings_dict[key] = value
            else:
                settings_dict[section_name] = section_config
        
        # Map schema fields to settings fields
        field_mapping = {
            "model": "llm_model",
            "api_key": "llm_api_key",
            "api_base": "llm_api_base",
            "temperature": "llm_temperature",
            "max_tokens": "llm_max_tokens"
        }
        
        for old_key, new_key in field_mapping.items():
            if old_key in settings_dict:
                settings_dict[new_key] = settings_dict.pop(old_key)
        
        settings = Settings()
        settings.load_from_dict(settings_dict)
        
        return settings
