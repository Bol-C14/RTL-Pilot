"""
Command-line interface for RTL-Pilot with LangChain integration.

This module provides a comprehensive CLI for running RTL verification workflows
using the new LangChain-based architecture.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import asyncio
import json

from ..config.settings import Settings
from ..workflows.default_flow import DefaultVerificationFlow
from ..agents.planner import VerificationPlanner


class CLIInterface:
    """
    Command-line interface for RTL-Pilot verification workflows with LangChain support.
    """
    
    def __init__(self):
        """Initialize the CLI interface with LangChain integration."""
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        
    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create the argument parser for the CLI.
        
        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description="RTL-Pilot: Automated RTL verification using LangChain and LLM agents",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Generate testbench for a simple design using LangChain
  rtl-pilot generate-tb --rtl my_design.v --output ./testbenches
  
  # Run complete verification workflow with LangChain orchestration
  rtl-pilot verify --rtl my_design.v --output ./verification --coverage 90
  
  # Run simulation using LangChain tools
  rtl-pilot simulate --rtl my_design.v --testbench my_tb.sv --output ./sim_results
  
  # Evaluate existing simulation results with LangChain analysis
  rtl-pilot evaluate --results ./sim_results/simulation.json --output ./evaluation
  
  # Chat with RTL verification agent
  rtl-pilot chat --rtl my_design.v
            """
        )
        
        # Global options
        parser.add_argument("--verbose", "-v", action="store_true",
                          help="Enable verbose logging")
        parser.add_argument("--config", type=Path,
                          help="Path to configuration file")
        parser.add_argument("--output", "-o", type=Path, default=Path("./rtl_pilot_output"),
                          help="Output directory for results")
        parser.add_argument("--provider", type=str, choices=["openai", "anthropic", "local"],
                          help="LLM provider to use (overrides config)")
        parser.add_argument("--model", type=str,
                          help="LLM model to use (overrides config)")
        
        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Generate testbench command
        tb_parser = subparsers.add_parser("generate-tb", help="Generate testbench using LangChain tools")
        tb_parser.add_argument("--rtl", type=Path, required=True,
                             help="Path to RTL source file")
        tb_parser.add_argument("--scenarios", type=Path,
                             help="Custom test scenarios file (JSON)")
        tb_parser.add_argument("--template", type=str, default="verilog_tb.jinja2",
                             help="Testbench template to use")
        tb_parser.add_argument("--style", type=str, choices=["simple", "uvm", "cocotb"], default="simple",
                             help="Testbench style")
        
        # Simulation command
        sim_parser = subparsers.add_parser("simulate", help="Run simulation using LangChain orchestration")
        sim_parser.add_argument("--rtl", type=Path, required=True,
                               help="Path to RTL source file")
        sim_parser.add_argument("--testbench", type=Path,
                               help="Path to testbench file (if not provided, will generate)")
        sim_parser.add_argument("--sim-time", default="1us",
                               help="Simulation time (e.g., 1us, 10ns)")
        sim_parser.add_argument("--vivado-project", type=Path,
                               help="Existing Vivado project file")
        sim_parser.add_argument("--coverage", action="store_true",
                               help="Enable coverage collection")
        
        # Evaluation command
        eval_parser = subparsers.add_parser("evaluate", help="Evaluate simulation results using LangChain analysis")
        eval_parser.add_argument("--results", type=Path, required=True,
                               help="Path to simulation results file")
        eval_parser.add_argument("--baseline", type=Path,
                               help="Baseline results for comparison")
        eval_parser.add_argument("--golden", type=Path,
                               help="Golden reference for functional verification")
        eval_parser.add_argument("--generate-feedback", action="store_true",
                               help="Generate improvement feedback using LangChain")
        
        # Complete verification workflow command
        verify_parser = subparsers.add_parser("verify", help="Run complete verification workflow with LangChain")
        verify_parser.add_argument("--rtl", type=Path, required=True,
                                 help="Path to RTL source file")
        verify_parser.add_argument("--coverage", type=float, default=90.0,
                                 help="Target coverage percentage")
        verify_parser.add_argument("--max-iterations", type=int, default=5,
                                 help="Maximum verification iterations")
        verify_parser.add_argument("--goals", type=Path,
                                 help="Verification goals file (JSON)")
        verify_parser.add_argument("--streaming", action="store_true",
                                 help="Enable streaming responses from LangChain")
        
        # Chat command (new)
        chat_parser = subparsers.add_parser("chat", help="Interactive chat with RTL verification agent")
        chat_parser.add_argument("--rtl", type=Path,
                                help="Path to RTL source file for context")
        chat_parser.add_argument("--project", type=Path,
                                help="Path to verification project directory")
        
        # Configuration commands
        config_parser = subparsers.add_parser("config", help="Configuration management")
        config_subparsers = config_parser.add_subparsers(dest="config_action")
        
        config_subparsers.add_parser("show", help="Show current configuration")
        init_parser = config_subparsers.add_parser("init", help="Initialize default configuration")
        init_parser.add_argument("--force", action="store_true",
                               help="Overwrite existing configuration")
        
        return parser
                                 help="Target coverage percentage")
        verify_parser.add_argument("--max-iterations", type=int, default=5,
                                 help="Maximum verification iterations")
        verify_parser.add_argument("--goals", type=Path,
                                 help="Verification goals file (JSON)")
        
        # Configuration commands
        config_parser = subparsers.add_parser("config", help="Configuration management")
        config_subparsers = config_parser.add_subparsers(dest="config_action")
        
        config_subparsers.add_parser("show", help="Show current configuration")
        init_parser = config_subparsers.add_parser("init", help="Initialize default configuration")
        init_parser.add_argument("--force", action="store_true",
                               help="Overwrite existing configuration")
        
        return parser
    
    def setup_logging(self, verbose: bool = False):
        """
        Setup logging configuration.
        
        Args:
            verbose: Enable debug level logging
        """
        level = logging.DEBUG if verbose else logging.INFO
        
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("rtl_pilot.log")
            ]
        )
    
    async def cmd_generate_tb(self, args: argparse.Namespace) -> int:
        """
        Handle testbench generation command.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            from ..agents.testbench_gen import RTLTestbenchGenerator
            
            self.logger.info(f"Generating testbench for {args.rtl}")
            
            # Initialize generator
            generator = RTLTestbenchGenerator(self.settings)
            
            # Load custom scenarios if provided
            test_scenarios = None
            if args.scenarios:
                with open(args.scenarios, 'r') as f:
                    test_scenarios = json.load(f)
            
            # Generate testbench using new async interface
            tb_result = await generator.generate_testbench_async(
                rtl_file=str(args.rtl),
                output_dir=str(args.output),
                test_scenarios=test_scenarios,
                style=args.style,
                template=args.template
            )
            
            if tb_result and tb_result.get('success', False):
                tb_file = tb_result.get('testbench_file')
                print(f"✓ Testbench generated: {tb_file}")
                print(f"  Style: {args.style}")
                print(f"  Template: {args.template}")
                if 'analysis' in tb_result:
                    analysis = tb_result['analysis']
                    print(f"  Module: {analysis.get('module_name', 'unknown')}")
                    print(f"  Test cases: {len(analysis.get('test_scenarios', []))}")
                return 0
            else:
                error_msg = tb_result.get('error', 'Unknown error') if tb_result else 'Generation failed'
                print(f"✗ Error: {error_msg}")
                return 1
            
        except Exception as e:
            self.logger.error(f"Testbench generation failed: {e}")
            print(f"✗ Error: {e}")
            return 1
    
    async def cmd_simulate(self, args: argparse.Namespace) -> int:
        """
        Handle simulation command.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            from ..agents.sim_runner import SimulationRunner
            from ..agents.testbench_gen import RTLTestbenchGenerator
            
            self.logger.info(f"Running simulation for {args.rtl}")
            
            # Initialize simulation runner
            runner = SimulationRunner(self.settings)
            
            # Generate testbench if not provided
            testbench_file = args.testbench
            if not testbench_file:
                self.logger.info("No testbench provided, generating one...")
                generator = RTLTestbenchGenerator(self.settings)
                tb_result = await generator.generate_testbench_async(
                    rtl_file=str(args.rtl),
                    output_dir=str(args.output)
                )
                if tb_result and tb_result.get('success', False):
                    testbench_file = Path(tb_result['testbench_file'])
                else:
                    print("✗ Failed to generate testbench")
                    return 1
            
            # Setup simulation project
            project_result = await runner.setup_simulation_project_async(
                rtl_files=[str(args.rtl)],
                testbench_file=str(testbench_file),
                project_dir=str(args.output / "simulation")
            )
            
            if not project_result.get('success', False):
                print(f"✗ Failed to setup simulation project: {project_result.get('error')}")
                return 1
            
            # Run simulation
            sim_result = await runner.run_simulation_async(
                project_file=project_result['project_file'],
                simulation_time=args.sim_time,
                enable_coverage=args.coverage
            )
            
            # Save results
            results_file = args.output / "simulation_results.json"
            args.output.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(sim_result, f, indent=2)
            
            if sim_result.get("success", False):
                print("✓ Simulation completed successfully")
                print(f"  Duration: {sim_result.get('duration', 'unknown')}")
                if args.coverage and 'coverage' in sim_result:
                    print(f"  Coverage: {sim_result['coverage'].get('total_coverage', 0):.1f}%")
                print(f"✓ Results saved to: {results_file}")
            else:
                print("✗ Simulation failed")
                for error in sim_result.get("errors", []):
                    print(f"  Error: {error}")
                if 'log_analysis' in sim_result:
                    print(f"  Log analysis: {sim_result['log_analysis'].get('summary', 'No summary')}")
            
            return 0 if sim_result.get("success", False) else 1
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            print(f"✗ Error: {e}")
            return 1
    
    async def cmd_evaluate(self, args: argparse.Namespace) -> int:
        """
        Handle evaluation command.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            from ..agents.evaluation import ResultEvaluator
            
            self.logger.info(f"Evaluating results from {args.results}")
            
            # Initialize evaluator
            evaluator = ResultEvaluator(self.settings)
            
            # Load simulation results
            with open(args.results, 'r') as f:
                sim_results = json.load(f)
            
            # Load baseline if provided
            baseline_results = None
            if args.baseline:
                with open(args.baseline, 'r') as f:
                    baseline_results = json.load(f)
            
            # Load golden reference if provided
            golden_data = None
            if args.golden:
                with open(args.golden, 'r') as f:
                    golden_data = json.load(f)
            
            # Evaluate results using new async interface
            evaluation = await evaluator.evaluate_simulation_results_async(
                sim_results=sim_results,
                expected_results=baseline_results,
                golden_reference=golden_data
            )
            
            if not evaluation.get('success', False):
                print(f"✗ Evaluation failed: {evaluation.get('error', 'Unknown error')}")
                return 1
            
            # Save evaluation results
            eval_file = args.output / "evaluation_results.json"
            args.output.mkdir(parents=True, exist_ok=True)
            with open(eval_file, 'w') as f:
                json.dump(evaluation, f, indent=2)
            
            # Generate report and feedback if requested
            report_result = await evaluator.generate_evaluation_report_async(
                evaluation, 
                str(args.output / "evaluation_report.json")
            )
            
            feedback_file = None
            if args.generate_feedback:
                feedback_result = await evaluator.generate_improvement_feedback_async(evaluation)
                if feedback_result.get('success', False):
                    feedback_file = args.output / "improvement_feedback.txt"
                    with open(feedback_file, 'w') as f:
                        f.write(feedback_result['feedback'])
            
            # Print summary
            results = evaluation.get('results', {})
            print(f"✓ Evaluation completed")
            print(f"  Overall Score: {results.get('overall_score', 0):.1f}/100")
            print(f"  Status: {'PASS' if results.get('pass', False) else 'FAIL'}")
            print(f"  Functional: {'PASS' if results.get('functional_correctness', {}).get('pass', False) else 'FAIL'}")
            if 'coverage_analysis' in results:
                print(f"  Coverage: {results['coverage_analysis'].get('coverage_score', 0):.1f}%")
            print(f"✓ Results saved to: {eval_file}")
            if report_result.get('success', False):
                print(f"✓ Report saved to: {report_result['report_file']}")
            if feedback_file:
                print(f"✓ Feedback saved to: {feedback_file}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            print(f"✗ Error: {e}")
            return 1
    
    async def cmd_verify(self, args: argparse.Namespace) -> int:
        """
        Handle complete verification workflow command.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            self.logger.info(f"Starting verification workflow for {args.rtl}")
            
            # Initialize planner
            planner = VerificationPlanner(self.settings)
            
            # Load verification goals
            verification_goals = {
                "coverage_target": args.coverage,
                "max_iterations": args.max_iterations
            }
            
            if args.goals:
                with open(args.goals, 'r') as f:
                    custom_goals = json.load(f)
                    verification_goals.update(custom_goals)
            
            # Create verification plan
            plan = planner.create_verification_plan(
                rtl_files=[args.rtl],
                verification_goals=verification_goals
            )
            
            print(f"✓ Verification plan created")
            print(f"  Target Coverage: {verification_goals['coverage_target']}%")
            print(f"  Max Iterations: {verification_goals['max_iterations']}")
            
            # Execute verification workflow
            results = await planner.execute_verification_workflow(plan, args.output)
            
            # Generate final report
            report_file = planner.generate_verification_report(
                results, args.output / "verification_report.json"
            )
            
            # Print summary
            print(f"\n✓ Verification workflow completed")
            print(f"  Success: {'YES' if results.get('success', False) else 'NO'}")
            print(f"  Iterations: {results.get('iterations', 0)}")
            print(f"  Final Coverage: {results.get('final_metrics', {}).get('final_coverage', 0):.1f}%")
            print(f"✓ Report saved to: {report_file}")
            
            return 0 if results.get("success", False) else 1
            
        except Exception as e:
            self.logger.error(f"Verification workflow failed: {e}")
            print(f"✗ Error: {e}")
            return 1
    
    async def cmd_chat(self, args: argparse.Namespace) -> int:
        """
        Handle interactive chat command with RTL verification agent.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success)
        """
        try:
            from ..llm.agent import RTLAgent
            
            print("🤖 RTL-Pilot Interactive Chat")
            print("=" * 40)
            print("Welcome to the RTL verification assistant!")
            
            # Initialize RTL agent
            agent = RTLAgent(self.settings)
            
            # Load RTL context if provided
            rtl_context = ""
            if args.rtl:
                print(f"📄 Loading RTL context from: {args.rtl}")
                with open(args.rtl, 'r') as f:
                    rtl_content = f.read()
                rtl_context = f"Working with RTL file '{args.rtl}':\n```verilog\n{rtl_content}\n```\n\n"
                
                # Analyze RTL to provide context
                analysis_result = await agent.analyze_rtl(str(args.rtl))
                if analysis_result:
                    print(f"✓ RTL analysis complete")
                    print(f"  Module: {analysis_result.get('module_name', 'unknown')}")
                    print(f"  Ports: {len(analysis_result.get('ports', []))} total")
                    print(f"  Clock domains: {len(analysis_result.get('clock_domains', []))}")
                    rtl_context += f"RTL Analysis: {analysis_result}\n\n"
            
            # Check if project directory exists and load context
            if args.project and Path(args.project).exists():
                print(f"📁 Loading project context from: {args.project}")
                # Add project files to context
                project_files = list(Path(args.project).rglob("*.v")) + list(Path(args.project).rglob("*.sv"))
                if project_files:
                    rtl_context += f"Project files found: {[str(f) for f in project_files]}\n\n"
            
            print("\nType 'help' for available commands, 'quit' to exit")
            print("-" * 40)
            
            # Interactive chat loop
            conversation_history = []
            
            while True:
                try:
                    # Get user input
                    user_input = input("\n🧑‍💻 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye! Happy verification!")
                        break
                    elif user_input.lower() == 'help':
                        self._print_chat_help()
                        continue
                    elif user_input.lower() == 'clear':
                        conversation_history = []
                        print("🧹 Conversation history cleared")
                        continue
                    elif user_input.lower().startswith('analyze '):
                        # Analyze RTL file
                        rtl_file = user_input[8:].strip()
                        if Path(rtl_file).exists():
                            analysis_result = await agent.analyze_rtl(rtl_file)
                            print(f"🔍 RTL Analysis for {rtl_file}:")
                            print(json.dumps(analysis_result, indent=2))
                        else:
                            print(f"❌ File not found: {rtl_file}")
                        continue
                    elif user_input.lower().startswith('generate '):
                        # Generate testbench
                        rtl_file = user_input[9:].strip()
                        if Path(rtl_file).exists():
                            print(f"🔧 Generating testbench for {rtl_file}...")
                            tb_result = await agent.generate_testbench(rtl_file, style="simple")
                            if tb_result:
                                print("✅ Testbench generated successfully!")
                                if 'testbench_code' in tb_result:
                                    # Save testbench to file
                                    tb_file = f"{Path(rtl_file).stem}_tb.sv"
                                    with open(tb_file, 'w') as f:
                                        f.write(tb_result['testbench_code'])
                                    print(f"💾 Saved to: {tb_file}")
                            else:
                                print("❌ Failed to generate testbench")
                        else:
                            print(f"❌ File not found: {rtl_file}")
                        continue
                    
                    # Prepare message with context
                    message = rtl_context + user_input
                    
                    # Get response from agent
                    print("🤖 RTL Agent: ", end="", flush=True)
                    
                    response_text = ""
                    async for chunk in agent.chat_stream(message, conversation_history):
                        if chunk:
                            print(chunk, end="", flush=True)
                            response_text += chunk
                    
                    print()  # New line after streaming response
                    
                    # Update conversation history
                    conversation_history.append({"role": "user", "content": user_input})
                    conversation_history.append({"role": "assistant", "content": response_text})
                    
                    # Limit conversation history to prevent context overflow
                    if len(conversation_history) > 20:
                        conversation_history = conversation_history[-20:]
                
                except KeyboardInterrupt:
                    print("\n\n👋 Chat interrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error during chat: {e}")
                    self.logger.error(f"Chat error: {e}")
                    continue
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Chat initialization failed: {e}")
            print(f"✗ Error: {e}")
            return 1
    
    def _print_chat_help(self):
        """Print help information for chat commands."""
        help_text = """
Available Commands:
  help                    - Show this help message
  quit, exit, q          - Exit the chat
  clear                  - Clear conversation history
  analyze <file.v>       - Analyze RTL file and show structure
  generate <file.v>      - Generate testbench for RTL file
  
Interactive Features:
  - Ask questions about RTL design and verification
  - Request testbench generation with specific requirements
  - Get help with simulation and debugging
  - Discuss verification strategies and coverage goals
  
Examples:
  "How can I verify this ALU module?"
  "Generate a comprehensive testbench for this counter"
  "What coverage metrics should I track?"
  "Help me debug this simulation failure"
        """
        print(help_text)
    
    def cmd_config_show(self, args: argparse.Namespace) -> int:
        """Show current configuration."""
        print("Current RTL-Pilot Configuration:")
        print("=" * 40)
        
        config_dict = {
            "vivado_path": str(self.settings.vivado_path),
            "llm_model": self.settings.llm_model,
            "prompts_dir": str(self.settings.prompts_dir),
            "output_dir": str(self.settings.output_dir),
            "simulation_timeout": self.settings.simulation_timeout
        }
        
        for key, value in config_dict.items():
            print(f"{key:20}: {value}")
        
        return 0
    
    def cmd_config_init(self, args: argparse.Namespace) -> int:
        """Initialize default configuration."""
        config_file = Path("rtl_pilot_config.json")
        
        if config_file.exists() and not args.force:
            print(f"Configuration file already exists: {config_file}")
            print("Use --force to overwrite")
            return 1
        
        default_config = {
            "vivado_path": "/opt/Xilinx/Vivado/2024.1/bin/vivado",
            "llm_model": "gpt-4",
            "prompts_dir": "./rtl_pilot/prompts",
            "output_dir": "./rtl_pilot_output",
            "simulation_timeout": "10ms"
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"✓ Configuration initialized: {config_file}")
        print("Edit this file to customize your settings")
        
        return 0
    
    async def run(self, argv: Optional[List[str]] = None) -> int:
        """
        Run the CLI interface.
        
        Args:
            argv: Command line arguments (uses sys.argv if None)
            
        Returns:
            Exit code
        """
        parser = self.create_parser()
        args = parser.parse_args(argv)
        
        # Setup logging
        self.setup_logging(args.verbose)
        
        # Load custom config if provided
        if hasattr(args, 'config') and args.config:
            self.settings.load_from_file(args.config)
        
        # Override settings with command line arguments
        if hasattr(args, 'provider') and args.provider:
            self.settings.llm_provider = args.provider
        if hasattr(args, 'model') and args.model:
            self.settings.llm_model = args.model
        
        # Handle commands
        if args.command == "generate-tb":
            return await self.cmd_generate_tb(args)
        elif args.command == "simulate":
            return await self.cmd_simulate(args)
        elif args.command == "evaluate":
            return await self.cmd_evaluate(args)
        elif args.command == "verify":
            return await self.cmd_verify(args)
        elif args.command == "chat":
            return await self.cmd_chat(args)
        elif args.command == "config":
            if args.config_action == "show":
                return self.cmd_config_show(args)
            elif args.config_action == "init":
                return self.cmd_config_init(args)
        else:
            parser.print_help()
            return 1


def main():
    """Main entry point for CLI."""
    cli = CLIInterface()
    return asyncio.run(cli.run())


if __name__ == "__main__":
    sys.exit(main())
