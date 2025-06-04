"""
Default verification workflow implementation with LangChain integration.

This module provides the standard RTL verification workflow that combines
testbench generation, simulation, and evaluation in a cohesive process
using the new LangChain-based architecture.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import time

from ..config.settings import Settings
from ..agents.testbench_gen import RTLTestbenchGenerator
from ..agents.sim_runner import SimulationRunner
from ..agents.evaluation import ResultEvaluator
from ..agents.planner import VerificationPlanner
from ..llm.agent import RTLAgent


class DefaultVerificationFlow:
    """
    Default verification workflow that implements the standard
    testbench → simulation → evaluation → refinement loop
    using LangChain architecture.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the verification workflow with LangChain support.
        
        Args:
            settings: Global configuration settings
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents with LangChain integration
        self.tb_generator = RTLTestbenchGenerator(settings)
        self.sim_runner = SimulationRunner(settings)
        self.evaluator = ResultEvaluator(settings)
        self.planner = VerificationPlanner(settings)
        self.rtl_agent = RTLAgent(settings)
        
        # Workflow state
        self.current_iteration = 0
        self.workflow_history = []
        self.metrics = {}
        
    async def run_verification(self, 
                             rtl_file: Path,
                             output_dir: Path,
                             verification_goals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete verification workflow with LangChain orchestration.
        
        Args:
            rtl_file: Path to RTL source file
            output_dir: Directory for output files
            verification_goals: Optional verification goals
            
        Returns:
            Dictionary containing workflow results
        """
        self.logger.info(f"Starting LangChain-powered verification workflow for {rtl_file}")
        
        # Setup output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default verification goals
        if verification_goals is None:
            verification_goals = {
                "coverage_target": getattr(self.settings, 'default_coverage_target', 90.0),
                "max_iterations": getattr(self.settings, 'max_verification_iterations', 5),
                "functional_correctness": True,
                "enable_tool_calling": True
            }
        
        # Use RTL agent for initial analysis and planning
        self.logger.info("Analyzing RTL with LangChain tools...")
        rtl_analysis = await self.rtl_agent.analyze_rtl(str(rtl_file))
        
        # Create verification plan using LangChain planner
        plan = await self.planner.create_verification_plan_async(
            rtl_files=[str(rtl_file)],
            verification_goals=verification_goals,
            rtl_analysis=rtl_analysis
        )
        
        # Execute the workflow
        workflow_results = await self._execute_workflow_plan(plan, output_dir)
        
        # Generate final report
        report_file = output_dir / "verification_report.json"
        await self._generate_final_report_async(workflow_results, report_file)
        
        self.logger.info(f"LangChain verification workflow completed. Report: {report_file}")
        
        return workflow_results
    
    async def _execute_workflow_plan(self, 
                                   plan: Dict[str, Any],
                                   output_dir: Path) -> Dict[str, Any]:
        """
        Execute the verification plan.
        
        Args:
            plan: Verification plan to execute
            output_dir: Output directory
            
        Returns:
            Workflow execution results
        """
        results = {
            "plan": plan,
            "iterations": [],
            "success": False,
            "final_metrics": {},
            "start_time": time.time(),
            "end_time": None
        }
        
        rtl_files = plan["rtl_files"]
        success_criteria = plan["success_criteria"]
        max_iterations = success_criteria["max_iterations"]
        
        try:
            for iteration in range(max_iterations):
                self.current_iteration = iteration + 1
                
                self.logger.info(f"Starting iteration {self.current_iteration}/{max_iterations}")
                
                iteration_dir = output_dir / f"iteration_{self.current_iteration}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                
                # Run single iteration
                iteration_results = await self._run_single_iteration(
                    rtl_files, iteration_dir, iteration
                )
                
                results["iterations"].append(iteration_results)
                
                # Check if success criteria met
                if self._check_success_criteria(iteration_results, success_criteria):
                    self.logger.info("Success criteria met!")
                    results["success"] = True
                    break
                
                # Prepare for next iteration if not the last one
                if iteration < max_iterations - 1:
                    await self._prepare_next_iteration(iteration_results, iteration_dir)
            
            # Calculate final metrics
            results["final_metrics"] = self._calculate_final_metrics(results)
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            results["error"] = str(e)
        finally:
            results["end_time"] = time.time()
            results["total_runtime"] = results["end_time"] - results["start_time"]
        
        return results
    
    async def _run_single_iteration(self, 
                                  rtl_files: List[str],
                                  iteration_dir: Path,
                                  iteration: int) -> Dict[str, Any]:
        """
        Run a single verification iteration using LangChain agents.
        
        Args:
            rtl_files: RTL files to verify
            iteration_dir: Directory for this iteration
            iteration: Current iteration number
            
        Returns:
            Results from this iteration
        """
        iteration_results = {
            "iteration": iteration + 1,
            "start_time": time.time(),
            "phases": {}
        }
        
        try:
            # Phase 1: Testbench Generation with LangChain
            self.logger.info("Phase 1: Generating testbench with LangChain")
            tb_result = await self._phase_testbench_generation_async(
                rtl_files[0], iteration_dir, iteration
            )
            iteration_results["phases"]["testbench_generation"] = tb_result
            
            # Phase 2: Simulation with LangChain orchestration
            self.logger.info("Phase 2: Running simulation with LangChain")
            sim_result = await self._phase_simulation_async(
                rtl_files, tb_result.get("testbench_file"), iteration_dir
            )
            iteration_results["phases"]["simulation"] = sim_result
            
            # Phase 3: Evaluation with LangChain analysis
            self.logger.info("Phase 3: Evaluating results with LangChain")
            eval_result = await self._phase_evaluation_async(
                sim_result, iteration_dir
            )
            iteration_results["phases"]["evaluation"] = eval_result
            
            # Phase 4: Generate feedback using LangChain (if needed)
            max_iterations = getattr(self.settings, 'max_verification_iterations', 5)
            if not eval_result.get("pass", False) and iteration < max_iterations - 1:
                self.logger.info("Phase 4: Generating feedback with LangChain")
                feedback_result = await self._phase_feedback_generation_async(
                    eval_result, tb_result.get("testbench_file"), iteration_dir
                )
                iteration_results["phases"]["feedback"] = feedback_result
            
            iteration_results["success"] = eval_result.get("pass", False)
            
        except Exception as e:
            self.logger.error(f"Iteration {iteration + 1} failed: {e}")
            iteration_results["error"] = str(e)
            iteration_results["success"] = False
        finally:
            iteration_results["end_time"] = time.time()
            iteration_results["runtime"] = iteration_results["end_time"] - iteration_results["start_time"]
        
        return iteration_results
    
    async def _phase_testbench_generation_async(self, 
                                              rtl_file: str,
                                              output_dir: Path,
                                              iteration: int) -> Dict[str, Any]:
        """Execute testbench generation phase with LangChain."""
        try:
            # Load feedback from previous iteration if available
            feedback_file = output_dir.parent / f"iteration_{iteration}" / "feedback.json"
            feedback = None
            if feedback_file.exists() and iteration > 0:
                with open(feedback_file, 'r') as f:
                    feedback = json.load(f)
            
            # Generate or refine testbench using async interface
            if iteration == 0 or feedback is None:
                tb_result = await self.tb_generator.generate_testbench_async(
                    rtl_file=rtl_file,
                    output_dir=str(output_dir)
                )
            else:
                # Refine based on feedback using LangChain
                prev_tb_file = output_dir.parent / f"iteration_{iteration}" / f"{Path(rtl_file).stem}_tb.sv"
                tb_result = await self.tb_generator.refine_testbench_async(
                    str(prev_tb_file), feedback
                )
            
            if tb_result and tb_result.get('success', False):
                return {
                    "success": True,
                    "testbench_file": tb_result['testbench_file'],
                    "iteration": iteration,
                    "analysis": tb_result.get('analysis', {}),
                    "generation_details": tb_result.get('generation_details', {})
                }
            else:
                return {
                    "success": False,
                    "error": tb_result.get('error', 'Testbench generation failed') if tb_result else 'No result'
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _phase_simulation_async(self, 
                                    rtl_files: List[str],
                                    testbench_file: str,
                                    output_dir: Path) -> Dict[str, Any]:
        """Execute simulation phase with LangChain orchestration."""
        try:
            # Setup simulation project using async interface
            project_result = await self.sim_runner.setup_simulation_project_async(
                rtl_files=rtl_files,
                testbench_file=testbench_file,
                project_dir=str(output_dir / "simulation")
            )
            
            if not project_result.get('success', False):
                return {
                    "success": False,
                    "error": f"Project setup failed: {project_result.get('error', 'Unknown error')}"
                }
            
            # Run simulation using async interface
            sim_results = await self.sim_runner.run_simulation_async(
                project_file=project_result['project_file'],
                simulation_time=getattr(self.settings, 'default_sim_time', '1us'),
                enable_coverage=True
            )
            
            # Generate simulation report
            if sim_results.get('success', False):
                report_result = await self.sim_runner.generate_simulation_report_async(
                    sim_results, str(output_dir / "simulation_report.json")
                )
                sim_results["report_file"] = report_result.get('report_file')
            
            return sim_results
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _phase_evaluation_async(self, 
                                    sim_results: Dict[str, Any],
                                    output_dir: Path) -> Dict[str, Any]:
        """Execute evaluation phase with LangChain analysis."""
        try:
            # Evaluate simulation results using async interface
            eval_result = await self.evaluator.evaluate_simulation_results_async(
                sim_results=sim_results
            )
            
            if not eval_result.get('success', False):
                return {
                    "success": False,
                    "error": f"Evaluation failed: {eval_result.get('error', 'Unknown error')}"
                }
            
            # Generate evaluation report
            report_result = await self.evaluator.generate_evaluation_report_async(
                eval_result, str(output_dir / "evaluation_report.json")
            )
            
            # Extract evaluation results
            results = eval_result.get('results', {})
            return {
                "success": True,
                "pass": results.get('pass', False),
                "overall_score": results.get('overall_score', 0),
                "coverage_analysis": results.get('coverage_analysis', {}),
                "functional_correctness": results.get('functional_correctness', {}),
                "report_file": report_result.get('report_file')
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _phase_feedback_generation_async(self, 
                                             eval_results: Dict[str, Any],
                                             testbench_file: str,
                                             output_dir: Path) -> Dict[str, Any]:
        """Generate feedback for next iteration using LangChain."""
        try:
            # Generate improvement feedback using async interface
            feedback_result = await self.evaluator.generate_improvement_feedback_async(
                eval_results
            )
            
            if feedback_result.get('success', False):
                # Save feedback to file
                feedback_file = output_dir / "feedback.json"
                feedback_data = {
                    "evaluation_results": eval_results,
                    "feedback": feedback_result['feedback'],
                    "suggestions": feedback_result.get('suggestions', []),
                    "timestamp": time.time()
                }
                
                with open(feedback_file, 'w') as f:
                    json.dump(feedback_data, f, indent=2)
                
                return {
                    "success": True,
                    "feedback_file": str(feedback_file),
                    "feedback": feedback_result['feedback']
                }
            else:
                return {
                    "success": False,
                    "error": f"Feedback generation failed: {feedback_result.get('error', 'Unknown error')}"
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _phase_simulation(self, 
                              rtl_files: List[Path],
                              testbench_file: Path,
                              output_dir: Path) -> Dict[str, Any]:
        """Execute simulation phase."""
        try:
            # Setup simulation project
            project_file = self.sim_runner.setup_simulation_project(
                rtl_files=rtl_files,
                testbench_file=testbench_file,
                project_dir=output_dir / "simulation"
            )
            
            # Run simulation
            sim_results = self.sim_runner.run_simulation(
                project_file=project_file,
                simulation_time=self.settings.default_sim_time
            )
            
            # Generate simulation report
            report_file = self.sim_runner.generate_simulation_report(
                sim_results, output_dir / "simulation_report.json"
            )
            
            sim_results["report_file"] = report_file
            return sim_results
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _phase_evaluation(self, 
                              sim_results: Dict[str, Any],
                              output_dir: Path) -> Dict[str, Any]:
        """Execute evaluation phase."""
        try:
            # Evaluate simulation results
            evaluation = self.evaluator.evaluate_simulation_results(sim_results)
            
            # Generate evaluation report
            report_file = self.evaluator.generate_evaluation_report(
                evaluation, output_dir / "evaluation_report.json"
            )
            
            evaluation["report_file"] = report_file
            return evaluation
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _phase_feedback_generation(self, 
                                       evaluation: Dict[str, Any],
                                       testbench_file: Path,
                                       output_dir: Path) -> Dict[str, Any]:
        """Execute feedback generation phase."""
        try:
            # Generate feedback for improvement
            feedback = self.evaluator.generate_feedback(evaluation, testbench_file)
            
            # Save feedback for next iteration
            feedback_file = output_dir / "feedback.json"
            with open(feedback_file, 'w') as f:
                json.dump(feedback, f, indent=2)
            
            return {
                "success": True,
                "feedback": feedback,
                "feedback_file": feedback_file
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _check_success_criteria(self, 
                               iteration_results: Dict[str, Any],
                               success_criteria: Dict[str, Any]) -> bool:
        """
        Check if success criteria are met.
        
        Args:
            iteration_results: Results from current iteration
            success_criteria: Success criteria to check
            
        Returns:
            True if success criteria are met
        """
        # Check if iteration was successful
        if not iteration_results.get("success", False):
            return False
        
        # Get evaluation results
        evaluation = iteration_results.get("phases", {}).get("evaluation", {})
        
        # Check overall pass status
        if not evaluation.get("pass", False):
            return False
        
        # Check coverage requirements
        coverage_analysis = evaluation.get("coverage_analysis", {})
        coverage_score = coverage_analysis.get("coverage_score", 0.0)
        min_coverage = success_criteria.get("minimum_coverage", 90.0)
        
        if coverage_score < min_coverage:
            return False
        
        # Check functional correctness if required
        if success_criteria.get("functional_correctness", True):
            functional = evaluation.get("functional_correctness", {})
            if not functional.get("all_tests_passed", True):
                return False
        
        return True
    
    async def _prepare_next_iteration(self, 
                                    iteration_results: Dict[str, Any],
                                    iteration_dir: Path) -> None:
        """
        Prepare for the next iteration based on current results.
        
        Args:
            iteration_results: Results from current iteration
            iteration_dir: Directory for current iteration
        """
        # Save iteration summary
        summary_file = iteration_dir / "iteration_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(iteration_results, f, indent=2, default=str)
        
        # Update workflow history
        self.workflow_history.append(iteration_results)
        
        # Log iteration summary
        success = iteration_results.get("success", False)
        runtime = iteration_results.get("runtime", 0)
        
        self.logger.info(f"Iteration {self.current_iteration} completed:")
        self.logger.info(f"  Success: {success}")
        self.logger.info(f"  Runtime: {runtime:.2f}s")
        
        if not success:
            evaluation = iteration_results.get("phases", {}).get("evaluation", {})
            score = evaluation.get("overall_score", 0)
            coverage = evaluation.get("coverage_analysis", {}).get("coverage_score", 0)
            
            self.logger.info(f"  Score: {score:.1f}/100")
            self.logger.info(f"  Coverage: {coverage:.1f}%")
    
    def _calculate_final_metrics(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate final verification metrics.
        
        Args:
            workflow_results: Complete workflow results
            
        Returns:
            Dictionary containing final metrics
        """
        iterations = workflow_results.get("iterations", [])
        
        if not iterations:
            return {}
        
        # Get final iteration results
        final_iteration = iterations[-1]
        final_evaluation = final_iteration.get("phases", {}).get("evaluation", {})
        
        metrics = {
            "total_iterations": len(iterations),
            "final_coverage": final_evaluation.get("coverage_analysis", {}).get("coverage_score", 0.0),
            "final_score": final_evaluation.get("overall_score", 0.0),
            "verification_passed": workflow_results.get("success", False),
            "total_runtime": workflow_results.get("total_runtime", 0.0),
            "average_iteration_time": 0.0,
            "total_test_cases": 0,
            "total_errors_found": 0
        }
        
        # Calculate average iteration time
        iteration_times = [it.get("runtime", 0) for it in iterations]
        if iteration_times:
            metrics["average_iteration_time"] = sum(iteration_times) / len(iteration_times)
        
        # Count total test cases and errors across all iterations
        for iteration in iterations:
            sim_results = iteration.get("phases", {}).get("simulation", {})
            metrics["total_errors_found"] += len(sim_results.get("errors", []))
        
        return metrics
    
    def _generate_final_report(self, 
                             workflow_results: Dict[str, Any],
                             report_file: Path) -> None:
        """
        Generate final verification report.
        
        Args:
            workflow_results: Complete workflow results
            report_file: Path to save the report
        """
        report = {
            "summary": {
                "verification_passed": workflow_results.get("success", False),
                "total_iterations": len(workflow_results.get("iterations", [])),
                "total_runtime": workflow_results.get("total_runtime", 0.0),
                "final_metrics": workflow_results.get("final_metrics", {})
            },
            "workflow_results": workflow_results,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rtl_pilot_version": "0.1.0"
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Final verification report saved: {report_file}")
    
    async def _generate_final_report_async(self, 
                                         workflow_results: Dict[str, Any],
                                         report_file: Path) -> None:
        """
        Generate final verification report with LangChain analysis.
        
        Args:
            workflow_results: Complete workflow results
            report_file: Path to save the report
        """
        # Use RTL agent to generate intelligent summary
        try:
            summary_prompt = f"""
            Analyze the following verification workflow results and provide a comprehensive summary:
            
            Results: {json.dumps(workflow_results, indent=2, default=str)}
            
            Please provide:
            1. Overall verification status and quality assessment
            2. Key findings and insights
            3. Coverage analysis summary
            4. Recommendations for improvement
            5. Risk assessment
            """
            
            summary_response = await self.rtl_agent.chat(summary_prompt, [])
            intelligent_summary = summary_response.get('response', 'Summary generation failed') if summary_response.get('success') else 'Summary generation failed'
        except Exception as e:
            intelligent_summary = f"Summary generation error: {e}"
        
        report = {
            "summary": {
                "verification_passed": workflow_results.get("success", False),
                "total_iterations": len(workflow_results.get("iterations", [])),
                "total_runtime": workflow_results.get("total_runtime", 0.0),
                "final_metrics": workflow_results.get("final_metrics", {}),
                "intelligent_summary": intelligent_summary
            },
            "workflow_results": workflow_results,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rtl_pilot_version": "0.1.0",
            "langchain_enabled": True,
            "llm_provider": self.settings.llm_provider,
            "llm_model": self.settings.llm_model
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"LangChain-powered verification report saved: {report_file}")

    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get current workflow status.
        
        Returns:
            Dictionary containing current status
        """
        return {
            "current_iteration": self.current_iteration,
            "total_iterations_run": len(self.workflow_history),
            "workflow_history": self.workflow_history,
            "current_metrics": self.metrics
        }
