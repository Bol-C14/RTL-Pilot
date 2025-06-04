"""
Verification workflow orchestrator and planner using LangChain architecture.

This module provides high-level planning and orchestration for complex RTL 
verification workflows using LangChain tools and agents.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import asyncio
import time

from ..config.settings import Settings
from .testbench_gen import RTLTestbenchGenerator
from .sim_runner import SimulationRunner
from .evaluation import ResultEvaluator
from ..llm.agent import RTLAgent
from ..llm.base import LLMResponse


class VerificationPhase(Enum):
    """Enumeration of verification workflow phases."""
    ANALYSIS = "analysis"
    TESTBENCH_GENERATION = "testbench_generation"
    SIMULATION = "simulation"
    EVALUATION = "evaluation"
    REFINEMENT = "refinement"
    COMPLETE = "complete"


class VerificationPlanner:
    """
    Orchestrates and plans complex RTL verification workflows using LangChain.
    Coordinates between different agents and manages verification campaigns.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the verification planner with LangChain integration.
        
        Args:
            settings: Global configuration settings
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.tb_generator = RTLTestbenchGenerator(settings)
        self.sim_runner = SimulationRunner(settings)
        self.evaluator = ResultEvaluator(settings)
        self.rtl_agent = RTLAgent(settings)
        
        # Workflow state
        self.current_phase = VerificationPhase.ANALYSIS
        self.workflow_history = []
        self.metrics_tracker = {}
        
    async def create_verification_plan_async(self, 
                                           rtl_files: List[Path],
                                           verification_goals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive verification plan using LangChain tools.
        
        Args:
            rtl_files: List of RTL files to verify
            verification_goals: Dictionary specifying verification objectives
            
        Returns:
            Dictionary containing the verification plan
        """
        self.logger.info("Creating verification plan using LangChain")
        
        # Prepare planning request for RTL agent
        planning_request = {
            "action": "create_verification_plan",
            "rtl_files": [str(f) for f in rtl_files],
            "verification_goals": verification_goals,
            "analyze_complexity": True,
            "estimate_effort": True,
            "plan_test_scenarios": True,
            "set_milestones": True
        }
        
        try:
            response = await self.rtl_agent.process_analysis_request(planning_request)
            
            # Extract structured plan from response
            plan = response.metadata.get("verification_plan", {})
            
            # Ensure required fields exist
            plan.setdefault("rtl_files", rtl_files)
            plan.setdefault("goals", verification_goals)
            plan.setdefault("phases", self._get_default_phases())
            plan.setdefault("estimated_duration", "TBD")
            plan.setdefault("resource_requirements", {})
            plan.setdefault("success_criteria", self._get_default_success_criteria(verification_goals))
            plan.setdefault("milestones", [])
            plan.setdefault("detailed_plan", response.content)
            
            # Parse phases from content if not in metadata
            if not plan["phases"] or len(plan["phases"]) == 0:
                plan["phases"] = self._extract_phases_from_content(response.content)
            
        except Exception as e:
            self.logger.error(f"LangChain planning failed, using fallback: {e}")
            plan = self._create_basic_plan_fallback(rtl_files, verification_goals)
        
        return plan
        
    def create_verification_plan(self, 
                               rtl_files: List[Path],
                               verification_goals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync wrapper for verification plan creation.
        
        Args:
            rtl_files: List of RTL files to verify
            verification_goals: Dictionary specifying verification objectives
            
        Returns:
            Dictionary containing the verification plan
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.create_verification_plan_async(rtl_files, verification_goals)
        )
    
    def _create_basic_plan_fallback(self, 
                                   rtl_files: List[Path],
                                   verification_goals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create basic verification plan as fallback.
        
        Args:
            rtl_files: List of RTL files to verify
            verification_goals: Dictionary specifying verification objectives
            
        Returns:
            Dictionary containing basic verification plan
        """
        plan = {
            "rtl_files": rtl_files,
            "goals": verification_goals,
            "phases": self._get_default_phases(),
            "estimated_duration": "4-6 hours",
            "resource_requirements": {"cpu_cores": 4, "memory_gb": 8},
            "success_criteria": self._get_default_success_criteria(verification_goals),
            "milestones": [
                {"phase": "analysis", "completion": "30 minutes"},
                {"phase": "testbench_generation", "completion": "1 hour"},
                {"phase": "simulation", "completion": "2-3 hours"},
                {"phase": "evaluation", "completion": "30 minutes"}
            ]
        }
        
        return plan
    
    def _get_default_phases(self) -> List[Dict[str, Any]]:
        """Get default verification phases."""
        return [
            {
                "name": "RTL Analysis",
                "phase": VerificationPhase.ANALYSIS.value,
                "duration": "30 minutes",
                "description": "Analyze RTL design and identify verification requirements"
            },
            {
                "name": "Testbench Generation", 
                "phase": VerificationPhase.TESTBENCH_GENERATION.value,
                "duration": "1 hour",
                "description": "Generate comprehensive testbenches using LangChain tools"
            },
            {
                "name": "Initial Simulation",
                "phase": VerificationPhase.SIMULATION.value,
                "duration": "2 hours", 
                "description": "Run simulation campaigns with generated testbenches"
            },
            {
                "name": "Result Evaluation",
                "phase": VerificationPhase.EVALUATION.value,
                "duration": "30 minutes",
                "description": "Evaluate results using LangChain analysis tools"
            },
            {
                "name": "Refinement",
                "phase": VerificationPhase.REFINEMENT.value,
                "duration": "1 hour",
                "description": "Refine testbenches based on LangChain feedback"
            }
        ]
    
    def _get_default_success_criteria(self, verification_goals: Dict[str, Any]) -> Dict[str, Any]:
        """Get default success criteria."""
        return {
            "minimum_coverage": verification_goals.get("coverage_target", 90.0),
            "functional_correctness": True,
            "performance_targets": verification_goals.get("performance", {}),
            "maximum_iterations": verification_goals.get("max_iterations", 5),
            "error_threshold": verification_goals.get("max_errors", 0),
            "warning_threshold": verification_goals.get("max_warnings", 5)
        }
    
    def _extract_phases_from_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract phases from LLM response content.
        
        Args:
            content: Response content from LLM
            
        Returns:
            List of phase dictionaries
        """
        # For now, return default phases
        # TODO: Implement intelligent phase extraction from content
        return self._get_default_phases()
    
    async def execute_verification_workflow(self, 
                                          verification_plan: Dict[str, Any],
                                          output_dir: Path) -> Dict[str, Any]:
        """
        Execute the complete verification workflow using LangChain orchestration.
        
        Args:
            verification_plan: Verification plan to execute
            output_dir: Directory for output files
            
        Returns:
            Dictionary containing workflow execution results
        """
        self.logger.info("Starting verification workflow execution with LangChain")
        
        workflow_results = {
            "plan": verification_plan,
            "execution_log": [],
            "phase_results": {},
            "final_metrics": {},
            "success": False,
            "iterations": 0,
            "langchain_enabled": True
        }
        
        try:
            rtl_files = verification_plan["rtl_files"]
            max_iterations = verification_plan["success_criteria"]["max_iterations"]
            
            for iteration in range(max_iterations):
                self.logger.info(f"Starting verification iteration {iteration + 1}")
                workflow_results["iterations"] = iteration + 1
                
                # Execute workflow phases using LangChain orchestration
                phase_results = await self._execute_workflow_iteration_async(
                    rtl_files, output_dir, iteration, verification_plan
                )
                
                workflow_results["phase_results"][f"iteration_{iteration + 1}"] = phase_results
                
                # Check if verification goals are met
                if self._check_success_criteria(phase_results, verification_plan["success_criteria"]):
                    workflow_results["success"] = True
                    self.logger.info("Verification goals achieved!")
                    break
                    
                # Use LangChain to plan next iteration if needed
                if iteration < max_iterations - 1:
                    self.logger.info("Planning next iteration using LangChain")
                    next_iteration_plan = await self._plan_next_iteration(
                        phase_results, verification_plan, iteration
                    )
                    workflow_results["execution_log"].append(
                        f"Iteration {iteration + 1} plan: {next_iteration_plan}"
                    )
                    
            # Generate final metrics using LangChain analysis
            workflow_results["final_metrics"] = await self._calculate_final_metrics_async(workflow_results)
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            workflow_results["execution_log"].append(f"ERROR: {e}")
            
        return workflow_results
    
    async def _execute_workflow_iteration_async(self, 
                                              rtl_files: List[Path],
                                              output_dir: Path,
                                              iteration: int,
                                              verification_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single iteration of the verification workflow using LangChain.
        
        Args:
            rtl_files: RTL files to verify
            output_dir: Output directory
            iteration: Current iteration number
            verification_plan: Original verification plan
            
        Returns:
            Dictionary containing iteration results
        """
        self.logger.info(f"Executing iteration {iteration + 1} with LangChain")
        
        iteration_results = {
            "iteration": iteration + 1,
            "phases_completed": [],
            "phase_timings": {},
            "overall_success": False,
            "metrics": {}
        }
        
        # Execute each phase using LangChain coordination
        for phase_info in verification_plan.get("phases", []):
            phase_name = phase_info["name"]
            phase_type = phase_info["phase"]
            
            self.logger.info(f"Executing phase: {phase_name}")
            phase_start = time.time()
            
            try:
                phase_result = await self._execute_phase_with_langchain(
                    phase_type, rtl_files, output_dir, iteration, phase_info
                )
                
                iteration_results["phases_completed"].append({
                    "phase": phase_name,
                    "success": phase_result.get("success", False),
                    "duration": time.time() - phase_start,
                    "result": phase_result
                })
                
                iteration_results["phase_timings"][phase_name] = time.time() - phase_start
                
                # Early exit if critical phase fails
                if not phase_result.get("success", False) and phase_type in ["testbench_generation", "simulation"]:
                    self.logger.warning(f"Critical phase {phase_name} failed, stopping iteration")
                    break
                    
            except Exception as e:
                self.logger.error(f"Phase {phase_name} failed: {e}")
                iteration_results["phases_completed"].append({
                    "phase": phase_name,
                    "success": False,
                    "duration": time.time() - phase_start,
                    "error": str(e)
                })
        
        # Calculate iteration success
        successful_phases = [p for p in iteration_results["phases_completed"] if p.get("success", False)]
        iteration_results["overall_success"] = len(successful_phases) >= len(verification_plan.get("phases", [])) * 0.8
        
        return iteration_results
    
    async def _execute_phase_with_langchain(self, 
                                           phase_type: str,
                                           rtl_files: List[Path],
                                           output_dir: Path,
                                           iteration: int,
                                           phase_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific workflow phase using LangChain coordination.
        
        Args:
            phase_type: Type of phase to execute
            rtl_files: RTL files being verified
            output_dir: Output directory
            iteration: Current iteration number
            phase_info: Phase configuration
            
        Returns:
            Dictionary containing phase execution results
        """
        phase_output_dir = output_dir / f"iteration_{iteration + 1}" / phase_type
        phase_output_dir.mkdir(parents=True, exist_ok=True)
        
        if phase_type == VerificationPhase.ANALYSIS.value:
            # RTL analysis using LangChain
            analysis_request = {
                "action": "analyze_rtl_design",
                "rtl_files": [str(f) for f in rtl_files],
                "analysis_depth": "comprehensive",
                "extract_interfaces": True,
                "identify_protocols": True
            }
            
            response = await self.rtl_agent.process_analysis_request(analysis_request)
            
            return {
                "success": response.success,
                "analysis_results": response.metadata,
                "analysis_report": response.content,
                "output_dir": str(phase_output_dir)
            }
            
        elif phase_type == VerificationPhase.TESTBENCH_GENERATION.value:
            # Testbench generation using LangChain (already implemented in tb_generator)
            try:
                testbench_results = await self.tb_generator.generate_testbench_async(
                    rtl_file=rtl_files[0],  # Primary RTL file
                    module_name=rtl_files[0].stem,
                    output_dir=phase_output_dir
                )
                
                return {
                    "success": True,
                    "testbench_file": testbench_results.get("testbench_file"),
                    "generation_log": testbench_results.get("generation_log"),
                    "output_dir": str(phase_output_dir)
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
                
        elif phase_type == VerificationPhase.SIMULATION.value:
            # Simulation using LangChain (already implemented in sim_runner)
            try:
                # Set up simulation project
                project_result = await self.sim_runner.setup_simulation_project_async(
                    rtl_files=rtl_files,
                    testbench_file=phase_output_dir / "testbench.sv",
                    project_dir=phase_output_dir / "sim_project"
                )
                
                if project_result["success"]:
                    # Run simulation
                    sim_results = await self.sim_runner.run_simulation_async(
                        project_file=project_result["project_file"],
                        simulation_time="1us"
                    )
                    
                    return {
                        "success": sim_results["success"],
                        "simulation_results": sim_results,
                        "output_dir": str(phase_output_dir)
                    }
                else:
                    return {"success": False, "error": "Project setup failed"}
                    
            except Exception as e:
                return {"success": False, "error": str(e)}
                
        elif phase_type == VerificationPhase.EVALUATION.value:
            # Evaluation using LangChain (already implemented in evaluator)
            try:
                # Get simulation results from previous phase
                sim_data_file = phase_output_dir.parent / "simulation" / "results.json"
                sim_results = {}
                
                if sim_data_file.exists():
                    with open(sim_data_file, 'r') as f:
                        sim_results = json.load(f)
                
                evaluation_results = await self.evaluator.evaluate_simulation_results_async(
                    sim_results=sim_results
                )
                
                return {
                    "success": True,
                    "evaluation_results": evaluation_results,
                    "output_dir": str(phase_output_dir)
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
                
        else:
            return {"success": False, "error": f"Unknown phase type: {phase_type}"}
    
    async def _plan_next_iteration(self, 
                                  phase_results: Dict[str, Any],
                                  verification_plan: Dict[str, Any],
                                  current_iteration: int) -> Dict[str, Any]:
        """
        Plan the next iteration using LangChain analysis.
        
        Args:
            phase_results: Results from current iteration
            verification_plan: Original verification plan
            current_iteration: Current iteration number
            
        Returns:
            Dictionary containing next iteration plan
        """
        # Prepare planning request for next iteration
        planning_request = {
            "action": "plan_next_iteration",
            "current_results": phase_results,
            "verification_plan": verification_plan,
            "iteration": current_iteration + 1,
            "analyze_gaps": True,
            "suggest_improvements": True
        }
        
        try:
            response = await self.rtl_agent.process_analysis_request(planning_request)
            
            # Extract planning recommendations from response
            next_plan = response.metadata.get("next_iteration_plan", {})
            next_plan.setdefault("recommendations", [])
            next_plan.setdefault("focus_areas", [])
            next_plan.setdefault("estimated_improvements", {})
            next_plan.setdefault("detailed_plan", response.content)
            
        except Exception as e:
            self.logger.error(f"LangChain iteration planning failed: {e}")
            next_plan = {
                "recommendations": ["Address simulation failures", "Improve test coverage"],
                "focus_areas": ["testbench_refinement", "coverage_enhancement"],
                "estimated_improvements": {"coverage": 10.0, "tests": 5},
                "error": str(e)
            }
        
        return next_plan
    
    async def _calculate_final_metrics_async(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate final verification metrics using LangChain analysis.
        
        Args:
            workflow_results: Complete workflow execution results
            
        Returns:
            Dictionary containing final metrics
        """
        # Prepare metrics calculation request
        metrics_request = {
            "action": "calculate_final_metrics",
            "workflow_results": workflow_results,
            "analyze_trends": True,
            "calculate_efficiency": True,
            "generate_summary": True
        }
        
        try:
            response = await self.rtl_agent.process_evaluation_request(metrics_request)
            
            # Extract metrics from response
            metrics = response.metadata.get("final_metrics", {})
            
            # Ensure required fields exist
            metrics.setdefault("total_iterations", workflow_results["iterations"])
            metrics.setdefault("final_coverage", 0.0)
            metrics.setdefault("total_test_cases", 0)
            metrics.setdefault("bugs_found", 0)
            metrics.setdefault("execution_time", "TBD")
            metrics.setdefault("efficiency_score", 0.0)
            metrics.setdefault("detailed_analysis", response.content)
            
        except Exception as e:
            self.logger.error(f"LangChain metrics calculation failed: {e}")
            metrics = self._calculate_basic_metrics_fallback(workflow_results)
        
        return metrics
    
    def _calculate_basic_metrics_fallback(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic metrics calculation fallback.
        
        Args:
            workflow_results: Complete workflow execution results
            
        Returns:
            Dictionary containing basic metrics
        """
        metrics = {
            "total_iterations": workflow_results["iterations"],
            "final_coverage": 0.0,
            "total_test_cases": 0,
            "bugs_found": 0,
            "execution_time": "TBD",
            "efficiency_score": 0.0
        }
        
        # Try to extract basic metrics from phase results
        for iteration_key, iteration_results in workflow_results.get("phase_results", {}).items():
            for phase in iteration_results.get("phases_completed", []):
                if phase.get("phase") == "Result Evaluation" and phase.get("success"):
                    eval_results = phase.get("result", {}).get("evaluation_results", {})
                    coverage_analysis = eval_results.get("coverage_analysis", {})
                    if coverage_analysis.get("coverage_score", 0) > metrics["final_coverage"]:
                        metrics["final_coverage"] = coverage_analysis.get("coverage_score", 0)
        
        return metrics
    
    def _check_success_criteria(self, 
                               phase_results: Dict[str, Any],
                               success_criteria: Dict[str, Any]) -> bool:
        """
        Check if the verification success criteria are met.
        
        Args:
            phase_results: Results from current workflow iteration
            success_criteria: Success criteria from verification plan
            
        Returns:
            True if success criteria are met
        """
        # Check overall iteration success
        if not phase_results.get("overall_success", False):
            return False
        
        # Extract evaluation results from completed phases
        evaluation_results = None
        for phase in phase_results.get("phases_completed", []):
            if phase.get("phase") == "Result Evaluation" and phase.get("success"):
                evaluation_results = phase.get("result", {}).get("evaluation_results", {})
                break
        
        if not evaluation_results:
            self.logger.warning("No evaluation results found")
            return False
            
        # Check overall pass status
        if not evaluation_results.get("pass", False):
            return False
            
        # Check coverage requirements
        coverage_analysis = evaluation_results.get("coverage_analysis", {})
        coverage_score = coverage_analysis.get("coverage_score", 0.0)
        min_coverage = success_criteria.get("minimum_coverage", 90.0)
        
        if coverage_score < min_coverage:
            self.logger.info(f"Coverage {coverage_score:.1f}% below target {min_coverage}%")
            return False
            
        # Check error threshold
        error_threshold = success_criteria.get("error_threshold", 0)
        if len(evaluation_results.get("issues_found", [])) > error_threshold:
            self.logger.info(f"Too many issues found: {len(evaluation_results.get('issues_found', []))}")
            return False
            
        return True
    
    async def generate_verification_report_async(self, 
                                               workflow_results: Dict[str, Any],
                                               output_file: Path) -> Path:
        """
        Generate comprehensive verification report using LangChain tools.
        
        Args:
            workflow_results: Complete workflow execution results
            output_file: Path to save report
            
        Returns:
            Path to generated report file
        """
        self.logger.info(f"Generating verification report: {output_file}")
        
        # Prepare report generation request
        report_request = {
            "action": "generate_verification_report",
            "workflow_results": workflow_results,
            "include_executive_summary": True,
            "include_detailed_analysis": True,
            "include_recommendations": True,
            "format": "comprehensive"
        }
        
        try:
            response = await self.rtl_agent.process_evaluation_request(report_request)
            
            # Generate comprehensive report
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if response.success and response.content:
                if output_file.suffix.lower() == '.json':
                    report_data = {
                        "workflow_summary": workflow_results,
                        "executive_summary": response.metadata.get("executive_summary", ""),
                        "detailed_analysis": response.content,
                        "recommendations": response.metadata.get("recommendations", []),
                        "final_metrics": response.metadata.get("final_metrics", {}),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "version": "2.0-langchain"
                    }
                    
                    import json
                    with open(output_file, 'w') as f:
                        json.dump(report_data, f, indent=2, default=str)
                else:
                    # For other formats, save content directly
                    with open(output_file, 'w') as f:
                        f.write(response.content)
            else:
                # Fallback report
                import json
                with open(output_file, 'w') as f:
                    json.dump(workflow_results, f, indent=2, default=str)
                    
        except Exception as e:
            self.logger.error(f"LangChain report generation failed: {e}")
            # Fallback report
            import json
            with open(output_file, 'w') as f:
                json.dump(workflow_results, f, indent=2, default=str)
            
        return output_file
        
    def generate_verification_report(self, 
                                   workflow_results: Dict[str, Any],
                                   output_file: Path) -> Path:
        """
        Sync wrapper for verification report generation.
        
        Args:
            workflow_results: Complete workflow execution results
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
            self.generate_verification_report_async(workflow_results, output_file)
        )
