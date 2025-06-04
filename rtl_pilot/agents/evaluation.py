"""
Result evaluation and analysis agent using LangChain architecture.

This module analyzes simulation results using LangChain tools and provides 
intelligent feedback for testbench refinement and design verification.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import re
import time
import asyncio

from ..config.settings import Settings
from ..utils.file_ops import read_file
from ..llm.agent import RTLAgent
from ..llm.base import LLMResponse


class ResultEvaluator:
    """
    Analyzes simulation results using LangChain tools and provides comprehensive 
    evaluation for RTL verification workflows.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the result evaluator with LangChain integration.
        
        Args:
            settings: Global configuration settings
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.rtl_agent = RTLAgent(settings)
        
    async def evaluate_simulation_results_async(self, 
                                              sim_results: Dict[str, Any],
                                              expected_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate simulation results using LangChain tools.
        
        Args:
            sim_results: Results from simulation run
            expected_results: Optional expected results for comparison
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info("Evaluating simulation results using LangChain")
        
        # Prepare evaluation request for RTL agent
        eval_request = {
            "action": "evaluate_simulation_results",
            "simulation_results": sim_results,
            "expected_results": expected_results,
            "evaluation_criteria": {
                "functional_correctness": True,
                "coverage_analysis": True,
                "performance_metrics": True,
                "error_analysis": True
            },
            "generate_recommendations": True
        }
        
        try:
            response = await self.rtl_agent.process_evaluation_request(eval_request)
            
            # Extract structured evaluation from response
            evaluation = response.metadata.get("evaluation_results", {})
            
            # Ensure required fields exist with defaults
            evaluation.setdefault("overall_score", 0.0)
            evaluation.setdefault("pass", False)
            evaluation.setdefault("functional_correctness", {})
            evaluation.setdefault("coverage_analysis", {})
            evaluation.setdefault("performance_metrics", {})
            evaluation.setdefault("recommendations", [])
            evaluation.setdefault("issues_found", [])
            evaluation.setdefault("detailed_analysis", response.content)
            
            # Parse recommendations from content if not in metadata
            if not evaluation["recommendations"] and response.content:
                evaluation["recommendations"] = self._extract_recommendations_from_content(response.content)
            
        except Exception as e:
            self.logger.error(f"LangChain evaluation failed, falling back to basic evaluation: {e}")
            evaluation = self._basic_evaluation_fallback(sim_results, expected_results)
        
        return evaluation
        
    def evaluate_simulation_results(self, 
                                   sim_results: Dict[str, Any],
                                   expected_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sync wrapper for simulation results evaluation.
        
        Args:
            sim_results: Results from simulation run
            expected_results: Optional expected results for comparison
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.evaluate_simulation_results_async(sim_results, expected_results)
        )
    
    def _basic_evaluation_fallback(self, 
                                  sim_results: Dict[str, Any],
                                  expected_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fallback evaluation when LangChain tools fail.
        
        Args:
            sim_results: Results from simulation run
            expected_results: Optional expected results for comparison
            
        Returns:
            Dictionary containing basic evaluation results
        """
        evaluation = {
            "overall_score": 0.0,
            "pass": False,
            "functional_correctness": {},
            "coverage_analysis": {},
            "performance_metrics": {},
            "recommendations": [],
            "issues_found": []
        }
        
        # Basic evaluation based on simulation success
        if sim_results.get("success", False):
            evaluation["overall_score"] += 30.0
            
        # Evaluate error count
        error_count = len(sim_results.get("errors", []))
        if error_count == 0:
            evaluation["overall_score"] += 40.0
        else:
            evaluation["issues_found"].extend([
                f"Simulation error: {error}" for error in sim_results.get("errors", [])
            ])
            
        # Evaluate warning count
        warning_count = len(sim_results.get("warnings", []))
        if warning_count == 0:
            evaluation["overall_score"] += 20.0
        elif warning_count < 5:
            evaluation["overall_score"] += 10.0
            
        # Coverage analysis
        coverage = sim_results.get("coverage", {})
        if coverage:
            evaluation["coverage_analysis"] = self._analyze_coverage(coverage)
            evaluation["overall_score"] += 10.0
        
        evaluation["pass"] = evaluation["overall_score"] >= 70.0
        
        return evaluation
    
    def _extract_recommendations_from_content(self, content: str) -> List[str]:
        """
        Extract recommendations from LLM response content.
        
        Args:
            content: Response content from LLM
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Look for common recommendation patterns
        patterns = [
            r"recommendation[s]?:?\s*(.+?)(?:\n|$)",
            r"suggest[s]?:?\s*(.+?)(?:\n|$)",
            r"improve[ment]?:?\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            recommendations.extend([match.strip() for match in matches if match.strip()])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _analyze_coverage(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze coverage metrics from simulation.
        
        Args:
            coverage_data: Coverage information from simulation
            
        Returns:
            Dictionary containing coverage analysis
        """
        analysis = {
            "line_coverage": coverage_data.get("line_coverage", 0.0),
            "branch_coverage": coverage_data.get("branch_coverage", 0.0),
            "toggle_coverage": coverage_data.get("toggle_coverage", 0.0),
            "fsm_coverage": coverage_data.get("fsm_coverage", 0.0),
            "uncovered_items": [],
            "coverage_score": 0.0
        }
        
        # Calculate overall coverage score
        coverages = [
            analysis["line_coverage"],
            analysis["branch_coverage"], 
            analysis["toggle_coverage"],
            analysis["fsm_coverage"]
        ]
        
        analysis["coverage_score"] = sum(c for c in coverages if c > 0) / len([c for c in coverages if c > 0]) if any(c > 0 for c in coverages) else 0.0
        
        return analysis
    
    async def analyze_functional_correctness_async(self, 
                                                  sim_results: Dict[str, Any],
                                                  golden_reference: Optional[Path] = None) -> Dict[str, Any]:
        """
        Analyze functional correctness using LangChain tools.
        
        Args:
            sim_results: Simulation results
            golden_reference: Optional golden reference for comparison
            
        Returns:
            Dictionary containing functional analysis
        """
        self.logger.info("Analyzing functional correctness using LangChain")
        
        # Prepare functional analysis request
        analysis_request = {
            "action": "analyze_functional_correctness",
            "simulation_results": sim_results,
            "golden_reference": str(golden_reference) if golden_reference else None,
            "analysis_depth": "comprehensive",
            "check_protocols": True,
            "verify_timing": True,
            "identify_bugs": True
        }
        
        try:
            response = await self.rtl_agent.process_evaluation_request(analysis_request)
            
            # Extract structured analysis from response
            analysis = response.metadata.get("functional_analysis", {})
            
            # Ensure required fields
            analysis.setdefault("functional_tests_passed", 0)
            analysis.setdefault("functional_tests_total", 0)
            analysis.setdefault("functional_score", 0.0)
            analysis.setdefault("mismatches", [])
            analysis.setdefault("protocol_violations", [])
            analysis.setdefault("timing_violations", [])
            analysis.setdefault("detailed_analysis", response.content)
            
        except Exception as e:
            self.logger.error(f"LangChain functional analysis failed: {e}")
            analysis = {
                "functional_tests_passed": 0,
                "functional_tests_total": 0,
                "functional_score": 0.0,
                "mismatches": [],
                "protocol_violations": [],
                "timing_violations": [],
                "error": str(e)
            }
        
        return analysis
    
    def analyze_functional_correctness(self, 
                                     sim_results: Dict[str, Any],
                                     golden_reference: Optional[Path] = None) -> Dict[str, Any]:
        """
        Sync wrapper for functional correctness analysis.
        
        Args:
            sim_results: Simulation results
            golden_reference: Optional golden reference for comparison
            
        Returns:
            Dictionary containing functional analysis
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.analyze_functional_correctness_async(sim_results, golden_reference)
        )
    
    async def generate_feedback_async(self, 
                                    evaluation_results: Dict[str, Any],
                                    original_testbench: Path) -> Dict[str, Any]:
        """
        Generate intelligent feedback using LangChain tools.
        
        Args:
            evaluation_results: Results from evaluation
            original_testbench: Path to original testbench
            
        Returns:
            Dictionary containing improvement feedback
        """
        self.logger.info("Generating improvement feedback using LangChain")
        
        # Read original testbench for context
        testbench_content = ""
        if original_testbench.exists():
            testbench_content = read_file(original_testbench)
        
        # Prepare feedback generation request
        feedback_request = {
            "action": "generate_improvement_feedback",
            "evaluation_results": evaluation_results,
            "testbench_content": testbench_content,
            "testbench_path": str(original_testbench),
            "feedback_types": [
                "improvement_suggestions",
                "additional_test_cases", 
                "coverage_improvements",
                "bug_fixes"
            ],
            "prioritize": True
        }
        
        try:
            response = await self.rtl_agent.process_evaluation_request(feedback_request)
            
            # Extract structured feedback from response
            feedback = response.metadata.get("feedback", {})
            
            # Ensure required fields
            feedback.setdefault("improvement_suggestions", [])
            feedback.setdefault("additional_test_cases", [])
            feedback.setdefault("coverage_improvements", [])
            feedback.setdefault("bug_fixes", [])
            feedback.setdefault("priority", "medium")
            feedback.setdefault("detailed_feedback", response.content)
            
            # Extract suggestions from content if not in metadata
            if not feedback["improvement_suggestions"] and response.content:
                feedback["improvement_suggestions"] = self._extract_suggestions_from_content(response.content)
            
        except Exception as e:
            self.logger.error(f"LangChain feedback generation failed, using fallback: {e}")
            feedback = self._basic_feedback_fallback(evaluation_results)
        
        return feedback
        
    def generate_feedback(self, 
                         evaluation_results: Dict[str, Any],
                         original_testbench: Path) -> Dict[str, Any]:
        """
        Sync wrapper for feedback generation.
        
        Args:
            evaluation_results: Results from evaluation
            original_testbench: Path to original testbench
            
        Returns:
            Dictionary containing improvement feedback
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.generate_feedback_async(evaluation_results, original_testbench)
        )
    
    def _basic_feedback_fallback(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic feedback generation fallback.
        
        Args:
            evaluation_results: Results from evaluation
            
        Returns:
            Dictionary containing basic feedback
        """
        feedback = {
            "improvement_suggestions": [],
            "additional_test_cases": [],
            "coverage_improvements": [],
            "bug_fixes": [],
            "priority": "medium"
        }
        
        # Basic feedback based on evaluation
        if not evaluation_results.get("pass", False):
            feedback["priority"] = "high"
            feedback["improvement_suggestions"].append(
                "Simulation failed - review errors and fix testbench issues"
            )
            
        coverage_score = evaluation_results.get("coverage_analysis", {}).get("coverage_score", 0.0)
        if coverage_score < 80.0:
            feedback["coverage_improvements"].append(
                f"Coverage is {coverage_score:.1f}% - add more test cases to improve coverage"
            )
            
        return feedback
    
    def _extract_suggestions_from_content(self, content: str) -> List[str]:
        """
        Extract improvement suggestions from LLM response content.
        
        Args:
            content: Response content from LLM
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        # Look for common suggestion patterns
        patterns = [
            r"suggestion[s]?:?\s*(.+?)(?:\n|$)",
            r"improve[ment]?[s]?:?\s*(.+?)(?:\n|$)",
            r"consider:?\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            suggestions.extend([match.strip() for match in matches if match.strip()])
        
        return suggestions[:10]  # Limit to top 10 suggestions
    
    async def compare_with_baseline_async(self, 
                                        current_results: Dict[str, Any],
                                        baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current results with baseline using LangChain tools.
        
        Args:
            current_results: Current evaluation results
            baseline_results: Baseline results for comparison
            
        Returns:
            Dictionary containing comparison analysis
        """
        self.logger.info("Comparing with baseline results using LangChain")
        
        # Prepare comparison request
        comparison_request = {
            "action": "compare_with_baseline",
            "current_results": current_results,
            "baseline_results": baseline_results,
            "comparison_metrics": [
                "overall_score",
                "coverage_metrics",
                "performance_metrics",
                "error_counts"
            ],
            "detect_regressions": True
        }
        
        try:
            response = await self.rtl_agent.process_evaluation_request(comparison_request)
            
            # Extract structured comparison from response
            comparison = response.metadata.get("comparison", {})
            
            # Ensure required fields
            comparison.setdefault("score_improvement", 0.0)
            comparison.setdefault("coverage_improvement", 0.0)
            comparison.setdefault("regression_detected", False)
            comparison.setdefault("improvements", [])
            comparison.setdefault("regressions", [])
            comparison.setdefault("detailed_comparison", response.content)
            
        except Exception as e:
            self.logger.error(f"LangChain comparison failed, using basic comparison: {e}")
            comparison = self._basic_baseline_comparison(current_results, baseline_results)
        
        return comparison
        
    def compare_with_baseline(self, 
                            current_results: Dict[str, Any],
                            baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sync wrapper for baseline comparison.
        
        Args:
            current_results: Current evaluation results
            baseline_results: Baseline results for comparison
            
        Returns:
            Dictionary containing comparison analysis
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.compare_with_baseline_async(current_results, baseline_results)
        )
    
    def _basic_baseline_comparison(self, 
                                  current_results: Dict[str, Any],
                                  baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic baseline comparison fallback.
        
        Args:
            current_results: Current evaluation results
            baseline_results: Baseline results for comparison
            
        Returns:
            Dictionary containing basic comparison
        """
        comparison = {
            "score_improvement": 0.0,
            "coverage_improvement": 0.0,
            "regression_detected": False,
            "improvements": [],
            "regressions": []
        }
        
        current_score = current_results.get("overall_score", 0.0)
        baseline_score = baseline_results.get("overall_score", 0.0)
        comparison["score_improvement"] = current_score - baseline_score
        
        if comparison["score_improvement"] < -5.0:
            comparison["regression_detected"] = True
            comparison["regressions"].append("Overall score decreased significantly")
            
        return comparison
    
    async def generate_evaluation_report_async(self, 
                                             evaluation_results: Dict[str, Any],
                                             output_file: Path) -> Path:
        """
        Generate comprehensive evaluation report using LangChain tools.
        
        Args:
            evaluation_results: Complete evaluation results
            output_file: Path to save report
            
        Returns:
            Path to generated report file
        """
        self.logger.info(f"Generating evaluation report: {output_file}")
        
        # Prepare report generation request
        report_request = {
            "action": "generate_evaluation_report",
            "evaluation_results": evaluation_results,
            "output_file": str(output_file),
            "include_visualizations": True,
            "include_recommendations": True,
            "include_summary": True,
            "format": "comprehensive"
        }
        
        try:
            response = await self.rtl_agent.process_evaluation_request(report_request)
            
            # Generate comprehensive report
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if response.success and response.content:
                if output_file.suffix.lower() == '.json':
                    report_data = {
                        "evaluation_summary": evaluation_results,
                        "detailed_analysis": response.content,
                        "recommendations": response.metadata.get("recommendations", []),
                        "visualizations": response.metadata.get("visualizations", {}),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "version": "2.0-langchain"
                    }
                    
                    with open(output_file, 'w') as f:
                        json.dump(report_data, f, indent=2)
                else:
                    # For other formats, save content directly
                    with open(output_file, 'w') as f:
                        f.write(response.content)
            else:
                # Fallback report
                report_content = {
                    "evaluation_summary": evaluation_results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "2.0-langchain",
                    "error": "Report generation failed"
                }
                
                with open(output_file, 'w') as f:
                    json.dump(report_content, f, indent=2)
                    
        except Exception as e:
            self.logger.error(f"LangChain report generation failed: {e}")
            # Fallback report
            report_content = {
                "evaluation_summary": evaluation_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "2.0-langchain",
                "error": str(e)
            }
            
            with open(output_file, 'w') as f:
                json.dump(report_content, f, indent=2)
            
        return output_file
        
    def generate_evaluation_report(self, 
                                  evaluation_results: Dict[str, Any],
                                  output_file: Path) -> Path:
        """
        Sync wrapper for evaluation report generation.
        
        Args:
            evaluation_results: Complete evaluation results
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
            self.generate_evaluation_report_async(evaluation_results, output_file)
        )
