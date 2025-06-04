"""
Vivado simulation runner for RTL verification using LangChain architecture.

This module handles running simulations in Vivado using LangChain tools
and collecting simulation results through the RTL agent framework.
"""

import logging
import subprocess
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import time
import asyncio

from ..config.settings import Settings
from ..utils.file_ops import read_file, write_file
from ..utils.vivado_interface import VivadoInterface
from ..llm.agent import RTLAgent
from ..llm.base import LLMResponse


class SimulationRunner:
    """
    Handles running RTL simulations using LangChain tools and RTL agent.
    Provides both async and sync interfaces for backward compatibility.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the simulation runner with LangChain integration.
        
        Args:
            settings: Global configuration settings
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.vivado = VivadoInterface(settings)
        self.rtl_agent = RTLAgent(settings)
        
    async def setup_simulation_project_async(self, 
                                           rtl_files: List[Path],
                                           testbench_file: Path,
                                           project_dir: Path) -> Dict[str, Any]:
        """
        Set up Vivado simulation project using LangChain tools.
        
        Args:
            rtl_files: List of RTL source files
            testbench_file: Testbench file
            project_dir: Directory for simulation project
            
        Returns:
            Dictionary containing project setup results
        """
        self.logger.info(f"Setting up simulation project in {project_dir}")
        
        # Use RTL agent to orchestrate project setup
        setup_request = {
            "action": "setup_simulation_project",
            "rtl_files": [str(f) for f in rtl_files],
            "testbench_file": str(testbench_file),
            "project_dir": str(project_dir),
            "simulator": "vivado",
            "requirements": {
                "create_project": True,
                "add_sources": True,
                "configure_simulation": True
            }
        }
        
        response = await self.rtl_agent.process_simulation_request(setup_request)
        
        return {
            "success": response.success,
            "project_file": project_dir / "sim_project.xpr",
            "setup_log": response.content,
            "metadata": response.metadata
        }
    
    def setup_simulation_project(self, 
                                rtl_files: List[Path],
                                testbench_file: Path,
                                project_dir: Path) -> Path:
        """
        Sync wrapper for simulation project setup.
        
        Args:
            rtl_files: List of RTL source files
            testbench_file: Testbench file
            project_dir: Directory for simulation project
            
        Returns:
            Path to project file
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            self.setup_simulation_project_async(rtl_files, testbench_file, project_dir)
        )
        
        if not result["success"]:
            raise RuntimeError(f"Project setup failed: {result['setup_log']}")
            
        return result["project_file"]
        
    def setup_simulation_project(self, 
                                rtl_files: List[Path],
                                testbench_file: Path,
                                project_dir: Path) -> Path:
        """
        Set up Vivado simulation project.
        
        Args:
            rtl_files: List of RTL source files
            testbench_file: Testbench file
            project_dir: Directory for simulation project
            
        Returns:
            Path to project file
        """
        self.logger.info(f"Setting up simulation project in {project_dir}")
    async def run_simulation_async(self, 
                                  project_file: Path,
                                  simulation_time: str = "1us",
                                  additional_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run simulation using LangChain tools and collect results.
        
        Args:
            project_file: Vivado project file
            simulation_time: Simulation run time
            additional_options: Additional simulation options
            
        Returns:
            Dictionary containing simulation results
        """
        self.logger.info(f"Running simulation for {simulation_time}")
        
        start_time = time.time()
        
        # Use RTL agent to run simulation
        sim_request = {
            "action": "run_simulation",
            "project_file": str(project_file),
            "simulation_time": simulation_time,
            "simulator": "vivado",
            "options": additional_options or {},
            "collect_coverage": True,
            "generate_waveforms": True
        }
        
        try:
            response = await self.rtl_agent.process_simulation_request(sim_request)
            
            sim_results = {
                "success": response.success,
                "runtime": time.time() - start_time,
                "errors": response.metadata.get("errors", []),
                "warnings": response.metadata.get("warnings", []),
                "coverage": response.metadata.get("coverage", {}),
                "waveform_file": response.metadata.get("waveform_file"),
                "log_file": response.metadata.get("log_file"),
                "simulation_output": response.content,
                "metadata": response.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            sim_results = {
                "success": False,
                "runtime": time.time() - start_time,
                "errors": [str(e)],
                "warnings": [],
                "coverage": {},
                "waveform_file": None,
                "log_file": None,
                "simulation_output": None,
                "metadata": {}
            }
        
        return sim_results
        
    def run_simulation(self, 
                      project_file: Path,
                      simulation_time: str = "1us",
                      additional_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sync wrapper for simulation execution.
        
        Args:
            project_file: Vivado project file
            simulation_time: Simulation run time
            additional_options: Additional simulation options
            
        Returns:
            Dictionary containing simulation results
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.run_simulation_async(project_file, simulation_time, additional_options)
        )
    
    async def analyze_simulation_logs_async(self, log_file: Path) -> Dict[str, Any]:
        """
        Analyze simulation log files using LangChain tools.
        
        Args:
            log_file: Path to simulation log file
            
        Returns:
            Dictionary containing log analysis results
        """
        self.logger.info(f"Analyzing simulation logs: {log_file}")
        
        if not log_file.exists():
            self.logger.warning(f"Log file not found: {log_file}")
            return {
                "total_errors": 0,
                "total_warnings": 0,
                "errors": [],
                "warnings": [],
                "performance_metrics": {},
                "coverage_summary": {}
            }
        
        # Use RTL agent for intelligent log analysis
        analysis_request = {
            "action": "analyze_logs",
            "log_file": str(log_file),
            "analysis_type": "comprehensive",
            "extract_metrics": True,
            "categorize_issues": True
        }
        
        response = await self.rtl_agent.process_evaluation_request(analysis_request)
        
        # Parse the response and extract structured data
        analysis = response.metadata.get("log_analysis", {})
        
        # Ensure required fields exist
        analysis.setdefault("total_errors", len(analysis.get("errors", [])))
        analysis.setdefault("total_warnings", len(analysis.get("warnings", [])))
        analysis.setdefault("errors", [])
        analysis.setdefault("warnings", [])
        analysis.setdefault("performance_metrics", {})
        analysis.setdefault("coverage_summary", {})
        
        return analysis
    
    def analyze_simulation_logs(self, log_file: Path) -> Dict[str, Any]:
        """
        Sync wrapper for simulation log analysis.
        
        Args:
            log_file: Path to simulation log file
            
        Returns:
            Dictionary containing log analysis results
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.analyze_simulation_logs_async(log_file))
    
    async def generate_simulation_report_async(self, 
                                             sim_results: Dict[str, Any],
                                             output_file: Path) -> Path:
        """
        Generate comprehensive simulation report using LangChain tools.
        
        Args:
            sim_results: Simulation results dictionary
            output_file: Path to save report
            
        Returns:
            Path to generated report file
        """
        self.logger.info(f"Generating simulation report: {output_file}")
        
        # Use RTL agent to generate comprehensive report
        report_request = {
            "action": "generate_simulation_report",
            "simulation_results": sim_results,
            "output_file": str(output_file),
            "include_visualizations": True,
            "include_recommendations": True,
            "format": "comprehensive"
        }
        
        response = await self.rtl_agent.process_evaluation_request(report_request)
        
        # Save the generated report
        if response.success and response.content:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.suffix.lower() == '.json':
                report_data = {
                    "simulation_summary": sim_results,
                    "analysis": response.content,
                    "recommendations": response.metadata.get("recommendations", []),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "2.0-langchain"
                }
                
                with open(output_file, 'w') as f:
                    json.dump(report_data, f, indent=2)
            else:
                # For other formats, save the content directly
                write_file(output_file, response.content)
        else:
            # Fallback to basic report
            report_content = {
                "simulation_summary": sim_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "2.0-langchain",
                "error": "Report generation failed"
            }
            
            with open(output_file, 'w') as f:
                json.dump(report_content, f, indent=2)
        
        return output_file
        
    def generate_simulation_report(self, 
                                  sim_results: Dict[str, Any],
                                  output_file: Path) -> Path:
        """
        Sync wrapper for simulation report generation.
        
        Args:
            sim_results: Simulation results dictionary
            output_file: Path to save report
            
        Returns:
            Path to generated report file
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.generate_simulation_report_async(sim_results, output_file)
        )
