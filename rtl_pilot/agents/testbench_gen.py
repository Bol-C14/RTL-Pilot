"""
LLM-based testbench generator for RTL designs using LangChain.

This module provides functionality to generate SystemVerilog/Verilog testbenches
using large language models with tool calling capabilities.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio

from ..config.settings import Settings
from ..utils.file_ops import read_file, write_file
from ..llm.agent import RTLAgent


class RTLTestbenchGenerator:
    """
    LLM-based testbench generator using LangChain with tool calling.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the testbench generator.
        
        Args:
            settings: Global configuration settings
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize RTL agent with LangChain
        self.agent = RTLAgent(
            llm_provider=getattr(settings, 'llm_provider', 'openai'),
            model_name=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )
        
    async def initialize(self) -> bool:
        """Initialize the generator and its components."""
        return await self.agent.initialize()
        
    async def generate_testbench(
        self,
        rtl_files: List[str],
        output_dir: str,
        top_module: Optional[str] = None,
        test_scenarios: Optional[List[str]] = None,
        testbench_style: str = "simple"
    ) -> Dict[str, Any]:
        """
        Generate testbench using LangChain agent with tool calling.
        
        Args:
            rtl_files: List of RTL file paths
            output_dir: Output directory for generated files
            top_module: Name of the top module
            test_scenarios: Test scenarios to implement
            testbench_style: Style of testbench to generate
            
        Returns:
            Dictionary with generation results
        """
        try:
            if not rtl_files:
                return {
                    "success": False,
                    "error": "No RTL files provided"
                }
            
            # Use the primary RTL file for analysis
            primary_rtl = rtl_files[0]
            
            # Run the complete verification workflow
            workflow_result = await self.agent.run_verification_workflow(
                rtl_file_path=primary_rtl,
                test_scenarios=test_scenarios,
                top_module=top_module,
                testbench_style=testbench_style
            )
            
            if not workflow_result.get("overall_success", False):
                return {
                    "success": False,
                    "error": "Verification workflow failed",
                    "details": workflow_result
                }
            
            # Extract testbench from workflow results
            testbench_step = None
            for step in workflow_result.get("workflow_steps", []):
                if step["step"] == "testbench_generation":
                    testbench_step = step["result"]
                    break
            
            if not testbench_step or not testbench_step.get("success", False):
                return {
                    "success": False,
                    "error": "Testbench generation failed in workflow"
                }
            
            # Write testbench to file
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            module_name = testbench_step.get("module_name", "testbench")
            testbench_file = output_path / f"tb_{module_name}.sv"
            
            testbench_code = testbench_step.get("testbench_code", "")
            write_file(testbench_file, testbench_code)
            
            return {
                "success": True,
                "testbench_file": str(testbench_file),
                "module_name": module_name,
                "testbench_style": testbench_style,
                "workflow_results": workflow_result,
                "recommendations": workflow_result.get("final_recommendations", [])
            }
            
        except Exception as e:
            self.logger.error(f"Testbench generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_rtl_design(self, rtl_file: Path) -> Dict[str, Any]:
        """
        Analyze RTL design (synchronous wrapper for async method).
        
        Args:
            rtl_file: Path to RTL source file
            
        Returns:
            Dictionary containing design analysis results
        """
        try:
            # Run async analysis
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.agent.analyze_rtl_design(str(rtl_file))
                    )
                    return future.result()
            else:
                return asyncio.run(self.agent.analyze_rtl_design(str(rtl_file)))
            
        except Exception as e:
            self.logger.error(f"RTL analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_verification_scenarios(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate verification scenarios based on analysis.
        
        Args:
            analysis: RTL analysis results
            
        Returns:
            List of verification scenario dictionaries
        """
        scenarios = []
        
        # Basic functionality scenario
        scenarios.append({
            "name": "basic_functionality",
            "description": "Test basic module functionality",
            "priority": "high"
        })
        
        # Add scenarios based on analysis
        if analysis.get("clock_domains"):
            scenarios.append({
                "name": "clock_domain_testing",
                "description": "Test clock domain interactions",
                "priority": "medium"
            })
        
        if analysis.get("state_machines"):
            scenarios.append({
                "name": "state_machine_coverage",
                "description": "Cover all state machine transitions",
                "priority": "high"
            })
        
        if analysis.get("reset_signals"):
            scenarios.append({
                "name": "reset_testing",
                "description": "Test reset behavior and recovery",
                "priority": "high"
            })
        
        verification_hints = analysis.get("verification_hints", [])
        for hint in verification_hints:
            if "boundary" in hint.lower():
                scenarios.append({
                    "name": "boundary_conditions",
                    "description": "Test input boundary conditions",
                    "priority": "medium"
                })
            if "timing" in hint.lower():
                scenarios.append({
                    "name": "timing_verification",
                    "description": "Verify timing constraints",
                    "priority": "medium"
                })
        
        return scenarios
    
    def generate_testbench_code(
        self, 
        analysis: Dict[str, Any], 
        scenarios: List[Dict[str, Any]]
    ) -> str:
        """
        Generate testbench code (synchronous wrapper).
        
        Args:
            analysis: RTL analysis results
            scenarios: Verification scenarios
            
        Returns:
            Generated testbench code
        """
        try:
            # Extract module info
            modules = analysis.get("modules", [])
            if not modules:
                return "// Error: No modules found in analysis"
            
            module_info = modules[0]  # Use first module
            scenario_names = [s["name"] for s in scenarios]
            
            # Run async testbench generation
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.agent.generate_testbench(module_info, scenario_names)
                    )
                    result = future.result()
            else:
                result = asyncio.run(
                    self.agent.generate_testbench(module_info, scenario_names)
                )
            
            return result.get("testbench_code", "// Error: Failed to generate testbench")
            
        except Exception as e:
            self.logger.error(f"Testbench code generation failed: {e}")
            return f"// Error: {e}"
    
    async def chat_with_agent(self, message: str) -> str:
        """
        Chat interface for interactive testbench generation.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        return await self.agent.chat(message)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of the underlying agent."""
        return self.agent.get_status()
        
        self.logger.info(f"Analyzing RTL design: {rtl_file}")
        
        analysis = {
            "module_name": "",
            "ports": [],
            "parameters": [],
            "clock_domains": [],
            "reset_signals": [],
            "interfaces": [],
            "state_machines": [],
            "complexity_score": 0
        }
        
        return analysis
    
    def generate_test_scenarios(self, design_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate test scenarios based on design analysis.
        
        Args:
            design_analysis: Results from RTL design analysis
            
        Returns:
            List of test scenario descriptions
        """
        # TODO: Implement LLM-based test scenario generation
        # - Basic functionality tests
        # - Edge case tests
        # - Performance tests
        # - Error injection tests
        
        self.logger.info("Generating test scenarios using LLM")
        
        scenarios = [
            {
                "name": "basic_functionality",
                "description": "Test basic module functionality",
                "priority": "high",
                "test_vectors": []
            },
            {
                "name": "edge_cases", 
                "description": "Test boundary conditions and edge cases",
                "priority": "medium",
                "test_vectors": []
            }
        ]
        
        return scenarios
    
    def generate_testbench(self, 
                          rtl_file: Path, 
                          output_dir: Path,
                          test_scenarios: Optional[List[Dict[str, Any]]] = None) -> Path:
        """
        Generate complete testbench for given RTL design.
        
        Args:
            rtl_file: Path to RTL source file
            output_dir: Directory to save generated testbench
            test_scenarios: Optional custom test scenarios
            
        Returns:
            Path to generated testbench file
        """
        self.logger.info(f"Generating testbench for {rtl_file}")
        
        # Analyze RTL design
        design_analysis = self.analyze_rtl_design(rtl_file)
        
        # Generate test scenarios if not provided
        if test_scenarios is None:
            test_scenarios = self.generate_test_scenarios(design_analysis)
        
        # Load testbench template
        template = self.template_env.get_template("verilog_tb.jinja2")
        
        # Generate testbench code
        testbench_code = template.render(
            design=design_analysis,
            scenarios=test_scenarios,
            settings=self.settings
        )
        
        # Save testbench file
        tb_file = output_dir / f"{design_analysis['module_name']}_tb.sv"
        write_file(tb_file, testbench_code)
        
        self.logger.info(f"Testbench generated: {tb_file}")
        return tb_file
    
    def refine_testbench(self, 
                        testbench_file: Path,
                        feedback: Dict[str, Any]) -> Path:
        """
        Refine testbench based on simulation feedback.
        
        Args:
            testbench_file: Path to existing testbench
            feedback: Feedback from simulation results
            
        Returns:
            Path to refined testbench file
        """
        # TODO: Implement feedback-based refinement
        # - Analyze simulation failures
        # - Generate additional test cases
        # - Fix testbench issues
        
        self.logger.info(f"Refining testbench based on feedback: {testbench_file}")
        
        return testbench_file
