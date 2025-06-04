"""
RTL Agent using LangChain with tool calling capabilities.

This module provides a high-level agent that can perform complete
RTL verification workflows using LLM tool calling.
"""

from typing import Dict, List, Any, Optional, Union
import asyncio
import logging
from pathlib import Path
import json

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable

from .base import BaseLLMInterface, LLMResponse
from .providers import OpenAIProvider, AnthropicProvider, LocalProvider
from .tools import RTLAnalysisTool, TestbenchGenerationTool, SimulationTool, EvaluationTool


class RTLAgent:
    """
    High-level RTL verification agent using LangChain with tool calling.
    
    This agent can:
    - Analyze RTL designs
    - Generate testbenches
    - Run simulations
    - Evaluate results
    - Provide recommendations
    """
    
    def __init__(
        self,
        settings_or_provider = None,
        model_name: str = "gpt-4",
        **llm_kwargs
    ):
        """
        Initialize the RTL agent.
        
        Args:
            settings_or_provider: Either a Settings object or LLM provider string
            model_name: Name of the model to use
            **llm_kwargs: Additional arguments for LLM initialization
        """
        self.logger = logging.getLogger(__name__)
        
        # Handle Settings object vs string provider
        if hasattr(settings_or_provider, 'llm_provider'):
            # It's a Settings object
            settings = settings_or_provider
            llm_provider = settings.llm_provider
            model_name = settings.llm_model
            llm_kwargs.update({
                'api_key': settings.llm_api_key,
                'api_base': settings.llm_api_base,
                'temperature': settings.llm_temperature,
                'max_tokens': settings.llm_max_tokens
            })
        else:
            # It's a provider string (backward compatibility)
            llm_provider = settings_or_provider or "openai"
        
        # Initialize LLM provider
        self.llm = self._create_llm_provider(llm_provider, model_name, **llm_kwargs)
        
        # Initialize tools
        self.tools = {
            "rtl_analysis": RTLAnalysisTool(),
            "testbench_generation": TestbenchGenerationTool(),
            "simulation": SimulationTool(),
            "evaluation": EvaluationTool()
        }
        
        # Register tools with LLM
        for tool in self.tools.values():
            self.llm.register_tool(tool)
        
        # Conversation history
        self.conversation_history: List[Any] = []
        
        # System prompt for RTL verification
        self.system_prompt = """
You are an expert RTL verification engineer with deep knowledge of:
- Digital design and Verilog/SystemVerilog
- Verification methodologies (UVM, directed tests, coverage-driven verification)
- Simulation tools (Vivado, ModelSim, QuestaSim)
- Coverage analysis and debugging

You have access to specialized tools for RTL verification workflows:
1. rtl_analysis: Analyze RTL designs and extract information
2. testbench_generation: Generate testbenches for RTL modules
3. run_simulation: Execute simulations with specified parameters
4. evaluate_results: Analyze simulation results and coverage

Your goal is to help users with complete RTL verification workflows by:
- Understanding their requirements
- Using appropriate tools in the correct sequence
- Providing expert analysis and recommendations
- Ensuring comprehensive verification coverage

Always use the available tools when appropriate, and provide clear explanations
of your analysis and recommendations.
"""
    
    def _create_llm_provider(
        self, 
        provider: str, 
        model_name: str, 
        **kwargs
    ) -> BaseLLMInterface:
        """Create the appropriate LLM provider."""
        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "local": LocalProvider
        }
        
        if provider not in providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        return providers[provider](model_name, **kwargs)
    
    async def initialize(self) -> bool:
        """
        Initialize the agent and its components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize LLM
            llm_initialized = await self.llm.initialize()
            if not llm_initialized:
                self.logger.error("Failed to initialize LLM")
                return False
            
            # Add system message to conversation
            self.conversation_history.append(
                SystemMessage(content=self.system_prompt)
            )
            
            self.logger.info("RTL Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RTL Agent: {e}")
            return False
    
    async def analyze_rtl_design(
        self, 
        rtl_file_path: str,
        top_module: Optional[str] = None,
        analysis_depth: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Analyze an RTL design file.
        
        Args:
            rtl_file_path: Path to the RTL file
            top_module: Name of the top module (optional)
            analysis_depth: Depth of analysis ("basic", "detailed", "comprehensive")
            
        Returns:
            Analysis results dictionary
        """
        try:
            query = f"""
Please analyze the RTL design file at: {rtl_file_path}

Requirements:
- Top module: {top_module or 'auto-detect'}
- Analysis depth: {analysis_depth}

Please use the rtl_analysis tool to perform this analysis and provide
a summary of the key findings including:
- Module interfaces and ports
- Clock domains and reset signals
- Key functional blocks
- Verification recommendations
"""
            
            response = await self._process_query(query)
            
            # Extract tool results from response
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "rtl_analysis":
                        tool_result = json.loads(tool_call.get("result", "{}"))
                        return tool_result
            
            return {
                "success": False,
                "error": "RTL analysis tool was not called or failed"
            }
            
        except Exception as e:
            self.logger.error(f"RTL analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_testbench(
        self,
        module_info: Dict[str, Any],
        test_scenarios: List[str],
        testbench_style: str = "simple",
        coverage_goals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a testbench for the analyzed RTL module.
        
        Args:
            module_info: Module information from RTL analysis
            test_scenarios: List of test scenarios to implement
            testbench_style: Style of testbench ("simple", "uvm", "cocotb")
            coverage_goals: Coverage requirements
            
        Returns:
            Testbench generation results
        """
        try:
            scenarios_text = ", ".join(test_scenarios)
            query = f"""
Please generate a testbench for the RTL module with the following specifications:

Module Information:
{json.dumps(module_info, indent=2)}

Test Scenarios: {scenarios_text}
Testbench Style: {testbench_style}
Coverage Goals: {json.dumps(coverage_goals or {}, indent=2)}

Please use the testbench_generation tool and provide:
- Complete testbench code
- Explanation of test scenarios
- Coverage strategy
- Any specific recommendations for this design
"""
            
            response = await self._process_query(query)
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "testbench_generation":
                        tool_result = json.loads(tool_call.get("result", "{}"))
                        return tool_result
            
            return {
                "success": False,
                "error": "Testbench generation tool was not called or failed"
            }
            
        except Exception as e:
            self.logger.error(f"Testbench generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def run_verification_workflow(
        self,
        rtl_file_path: str,
        test_scenarios: Optional[List[str]] = None,
        top_module: Optional[str] = None,
        testbench_style: str = "simple",
        simulator: str = "vivado"
    ) -> Dict[str, Any]:
        """
        Run a complete verification workflow.
        
        Args:
            rtl_file_path: Path to the RTL file
            test_scenarios: Test scenarios (auto-generated if None)
            top_module: Top module name
            testbench_style: Testbench style
            simulator: Simulator to use
            
        Returns:
            Complete workflow results
        """
        workflow_results = {
            "workflow_steps": [],
            "overall_success": True,
            "final_recommendations": []
        }
        
        try:
            # Step 1: Analyze RTL
            self.logger.info("Step 1: Analyzing RTL design...")
            analysis_result = await self.analyze_rtl_design(
                rtl_file_path, top_module, "comprehensive"
            )
            
            workflow_results["workflow_steps"].append({
                "step": "rtl_analysis",
                "result": analysis_result
            })
            
            if not analysis_result.get("success", False):
                workflow_results["overall_success"] = False
                return workflow_results
            
            # Extract module info for testbench generation
            modules = analysis_result.get("modules", [])
            if not modules:
                workflow_results["overall_success"] = False
                workflow_results["final_recommendations"].append("No modules found in RTL")
                return workflow_results
            
            target_module = modules[0]  # Use first module as target
            
            # Step 2: Generate test scenarios if not provided
            if test_scenarios is None:
                test_scenarios = await self._generate_test_scenarios(analysis_result)
            
            # Step 3: Generate testbench
            self.logger.info("Step 2: Generating testbench...")
            testbench_result = await self.generate_testbench(
                target_module, test_scenarios, testbench_style
            )
            
            workflow_results["workflow_steps"].append({
                "step": "testbench_generation",
                "result": testbench_result
            })
            
            if not testbench_result.get("success", False):
                workflow_results["overall_success"] = False
                return workflow_results
            
            # Step 4: Run simulation (simulated for now)
            self.logger.info("Step 3: Running simulation...")
            simulation_result = await self._run_simulation_step(
                testbench_result, [rtl_file_path], simulator
            )
            
            workflow_results["workflow_steps"].append({
                "step": "simulation",
                "result": simulation_result
            })
            
            # Step 5: Evaluate results
            self.logger.info("Step 4: Evaluating results...")
            evaluation_result = await self._evaluate_results_step(
                simulation_result, test_scenarios
            )
            
            workflow_results["workflow_steps"].append({
                "step": "evaluation",
                "result": evaluation_result
            })
            
            # Generate final recommendations
            workflow_results["final_recommendations"] = evaluation_result.get(
                "recommendations", []
            )
            
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Verification workflow failed: {e}")
            workflow_results["overall_success"] = False
            workflow_results["error"] = str(e)
            return workflow_results
    
    async def _process_query(self, query: str) -> LLMResponse:
        """Process a user query and return LLM response."""
        # Add user message to conversation
        self.conversation_history.append(HumanMessage(content=query))
        
        # Get LLM response with tools
        response = await self.llm.generate_response(
            self.conversation_history,
            tools=list(self.tools.values())
        )
        
        # Add response to conversation history
        if response.success:
            self.conversation_history.append(
                AIMessage(content=response.content)
            )
        
        return response
    
    async def _generate_test_scenarios(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate test scenarios based on RTL analysis."""
        scenarios = ["basic_functionality"]
        
        # Add scenarios based on analysis
        if analysis_result.get("clock_domains"):
            scenarios.append("clock_domain_crossing")
        
        if analysis_result.get("state_machines"):
            scenarios.append("state_machine_coverage")
        
        if analysis_result.get("reset_signals"):
            scenarios.append("reset_testing")
        
        verification_hints = analysis_result.get("verification_hints", [])
        for hint in verification_hints:
            if "boundary" in hint.lower():
                scenarios.append("boundary_conditions")
            if "timing" in hint.lower():
                scenarios.append("timing_verification")
        
        return scenarios
    
    async def _run_simulation_step(
        self,
        testbench_result: Dict[str, Any],
        rtl_files: List[str],
        simulator: str
    ) -> Dict[str, Any]:
        """Run simulation step of the workflow."""
        query = f"""
Please run a simulation with the following parameters:

Testbench: Generated testbench code
RTL Files: {rtl_files}
Simulator: {simulator}
Simulation Time: 1us

Use the run_simulation tool to execute this simulation.
"""
        
        response = await self._process_query(query)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "run_simulation":
                    return json.loads(tool_call.get("result", "{}"))
        
        return {"success": False, "error": "Simulation tool not called"}
    
    async def _evaluate_results_step(
        self,
        simulation_result: Dict[str, Any],
        test_scenarios: List[str]
    ) -> Dict[str, Any]:
        """Evaluate simulation results step."""
        query = f"""
Please evaluate the simulation results:

Simulation Results:
{json.dumps(simulation_result, indent=2)}

Test Scenarios: {test_scenarios}

Use the evaluate_results tool to analyze:
- Coverage metrics
- Test scenario completeness
- Any issues or failures
- Recommendations for improvement
"""
        
        response = await self._process_query(query)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "evaluate_results":
                    return json.loads(tool_call.get("result", "{}"))
        
        return {"success": False, "error": "Evaluation tool not called"}
    
    async def chat(self, message: str) -> str:
        """
        Chat interface for interactive use.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        try:
            response = await self._process_query(message)
            return response.content if response.success else f"Error: {response.error}"
            
        except Exception as e:
            self.logger.error(f"Chat failed: {e}")
            return f"Sorry, I encountered an error: {e}"
    
    def get_conversation_history(self) -> List[str]:
        """Get formatted conversation history."""
        history = []
        for message in self.conversation_history:
            if hasattr(message, 'type'):
                if message.type == 'human':
                    history.append(f"User: {message.content}")
                elif message.type == 'ai':
                    history.append(f"Agent: {message.content}")
                elif message.type == 'system':
                    history.append(f"System: {message.content}")
        return history
    
    def clear_conversation(self) -> None:
        """Clear conversation history (keeping system prompt)."""
        self.conversation_history = [
            SystemMessage(content=self.system_prompt)
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and information."""
        return {
            "llm_provider": self.llm.__class__.__name__,
            "model_info": self.llm.get_model_info(),
            "available_tools": list(self.tools.keys()),
            "conversation_length": len(self.conversation_history),
            "initialized": self.llm._model is not None
        }
    
    def get_available_tools(self) -> List[Any]:
        """
        Get list of available tools.
        
        Returns:
            List of available tool instances
        """
        return list(self.tools.values())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all available tools.
        
        Returns:
            Dictionary mapping tool names to descriptions
        """
        return {
            "rtl_analysis": "Analyze RTL designs and extract structural information",
            "testbench_generation": "Generate comprehensive testbenches for RTL modules",
            "run_simulation": "Execute simulations with customizable parameters",
            "evaluate_results": "Analyze simulation results and generate coverage reports"
        }
