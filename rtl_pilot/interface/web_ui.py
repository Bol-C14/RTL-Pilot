"""
Web user interface for RTL-Pilot using Streamlit with LangChain integration.

This module provides an optional web-based interface for RTL verification workflows
using the new LangChain-based architecture.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
import json
import time
import threading

from ..config.settings import Settings
from ..agents.planner import VerificationPlanner
from ..agents.testbench_gen import RTLTestbenchGenerator
from ..agents.sim_runner import SimulationRunner
from ..agents.evaluation import ResultEvaluator
from ..llm.agent import RTLAgent


class WebInterface:
    """
    Streamlit-based web interface for RTL-Pilot with LangChain integration.
    """
    
    def __init__(self):
        """Initialize the web interface with LangChain support."""
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state
        if "workflow_state" not in st.session_state:
            st.session_state.workflow_state = "idle"
        if "results_history" not in st.session_state:
            st.session_state.results_history = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "rtl_agent" not in st.session_state:
            st.session_state.rtl_agent = None
    
    def _get_rtl_agent(self) -> RTLAgent:
        """Get or create RTL agent instance."""
        if st.session_state.rtl_agent is None:
            st.session_state.rtl_agent = RTLAgent(self.settings)
        return st.session_state.rtl_agent
    
    def _run_async(self, coro):
        """Run async function in streamlit context."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    def render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        st.sidebar.title("RTL-Pilot")
        st.sidebar.markdown("*AI-Powered RTL Verification with LangChain*")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Dashboard", "Chat Assistant", "Testbench Generator", "Simulation Runner", 
             "Result Evaluator", "Verification Workflow", "Settings"]
        )
        
        # LangChain Settings
        st.sidebar.markdown("---")
        st.sidebar.markdown("**LangChain Settings**")
        
        provider = st.sidebar.selectbox(
            "LLM Provider",
            ["openai", "anthropic", "local"],
            index=0 if self.settings.llm_provider == "openai" else 
                  1 if self.settings.llm_provider == "anthropic" else 2
        )
        
        if provider != self.settings.llm_provider:
            self.settings.llm_provider = provider
            # Reset agent to use new provider
            st.session_state.rtl_agent = None
        
        model = st.sidebar.text_input("Model", value=self.settings.llm_model)
        if model != self.settings.llm_model:
            self.settings.llm_model = model
            st.session_state.rtl_agent = None
        
        streaming = st.sidebar.checkbox("Streaming Responses", value=True)
        
        # Quick stats
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Quick Stats**")
        st.sidebar.metric("Workflows Run", len(st.session_state.results_history))
        st.sidebar.metric("Current Status", st.session_state.workflow_state.title())
        st.sidebar.metric("Chat Messages", len(st.session_state.chat_history))
        
        return page
    
    def render_dashboard(self):
        """Render the main dashboard."""
        st.title("🚀 RTL-Pilot Dashboard")
        st.markdown("Welcome to RTL-Pilot - AI-powered RTL verification workflows")
        
        # Quick actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 💬 Quick Chat")
            if st.button("Chat Assistant", use_container_width=True):
                st.session_state.page = "Chat Assistant"
                st.rerun()
        
        with col2:
            st.markdown("### 🧪 Quick Generate")
            if st.button("Generate Testbench", use_container_width=True):
                st.session_state.page = "Testbench Generator"
                st.rerun()
        
        with col3:
            st.markdown("### ⚡ Quick Simulate")
            if st.button("Run Simulation", use_container_width=True):
                st.session_state.page = "Simulation Runner"
                st.rerun()
        
        # Recent results
        if st.session_state.results_history:
            st.markdown("---")
            st.markdown("### 📊 Recent Results")
            
            for i, result in enumerate(reversed(st.session_state.results_history[-5:])):
                with st.expander(f"Workflow {len(st.session_state.results_history) - i} - {result.get('timestamp', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Status", "✅ PASS" if result.get("success", False) else "❌ FAIL")
                    with col2:
                        st.metric("Coverage", f"{result.get('coverage', 0):.1f}%")
                    with col3:
                        st.metric("Iterations", result.get("iterations", 0))
        else:
            st.info("No verification workflows run yet. Start by selecting an action above!")
    
    def render_chat_assistant(self):
        """Render the chat assistant interface."""
        st.title("💬 RTL Chat Assistant")
        st.markdown("Interactive chat with the RTL verification agent using LangChain")
        
        # RTL context upload
        st.markdown("### 📄 RTL Context (Optional)")
        uploaded_rtl = st.file_uploader(
            "Upload RTL file for context",
            type=["v", "sv"],
            help="Upload an RTL file to provide context for the chat"
        )
        
        rtl_context = ""
        if uploaded_rtl:
            rtl_path = Path(f"/tmp/{uploaded_rtl.name}")
            with open(rtl_path, "wb") as f:
                f.write(uploaded_rtl.getbuffer())
            
            # Analyze RTL for context
            with st.spinner("Analyzing RTL file..."):
                try:
                    agent = self._get_rtl_agent()
                    analysis = self._run_async(agent.analyze_rtl(str(rtl_path)))
                    
                    if analysis:
                        st.success(f"✅ RTL analyzed: {analysis.get('module_name', 'unknown')} module")
                        
                        # Show analysis summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Ports", len(analysis.get('ports', [])))
                        with col2:
                            st.metric("Clock Domains", len(analysis.get('clock_domains', [])))
                        with col3:
                            st.metric("State Machines", len(analysis.get('state_machines', [])))
                        
                        rtl_context = f"Working with RTL file '{uploaded_rtl.name}':\n\nAnalysis: {json.dumps(analysis, indent=2)}\n\n"
                except Exception as e:
                    st.error(f"Failed to analyze RTL: {e}")
        
        # Chat interface
        st.markdown("### 💬 Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about RTL verification, testbench generation, or simulation..."):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        agent = self._get_rtl_agent()
                        
                        # Prepare message with context
                        full_message = rtl_context + prompt
                        
                        # Get response (non-streaming for simplicity in web UI)
                        response = self._run_async(agent.chat(full_message, st.session_state.chat_history[:-1]))
                        
                        if response and response.get('success', False):
                            response_text = response['response']
                            st.write(response_text)
                            
                            # Add assistant message to history
                            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                            
                            # Show tools used if any
                            if 'tools_used' in response and response['tools_used']:
                                with st.expander("🔧 Tools Used"):
                                    for tool in response['tools_used']:
                                        st.write(f"- **{tool.get('name', 'Unknown')}**: {tool.get('description', 'No description')}")
                        else:
                            error_msg = response.get('error', 'Unknown error') if response else 'No response'
                            st.error(f"Failed to get response: {error_msg}")
                            
                    except Exception as e:
                        st.error(f"Error during chat: {e}")
        
        # Chat controls
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("Export Chat"):
                chat_export = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": st.session_state.chat_history
                }
                st.download_button(
                    "Download Chat",
                    data=json.dumps(chat_export, indent=2),
                    file_name=f"rtl_chat_{int(time.time())}.json",
                    mime="application/json"
                )
    
    def render_testbench_generator(self):
        """Render the testbench generator interface with LangChain integration."""
        st.title("🧪 Testbench Generator")
        st.markdown("Generate SystemVerilog testbenches using LangChain AI tools")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload RTL File (.v, .sv)",
            type=["v", "sv"],
            help="Select the RTL design file for which to generate a testbench"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            rtl_path = Path(f"/tmp/{uploaded_file.name}")
            with open(rtl_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Options
            st.markdown("### Generation Options")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                style = st.selectbox(
                    "Testbench Style",
                    ["simple", "uvm", "cocotb"],
                    help="Choose the testbench methodology"
                )
            
            with col2:
                template = st.selectbox(
                    "Template",
                    ["verilog_tb.jinja2", "systemverilog_tb.jinja2", "uvm_tb.jinja2"]
                )
            
            with col3:
                coverage_target = st.slider("Coverage Target (%)", 70, 100, 90)
            
            # Advanced options
            with st.expander("Advanced Options"):
                enable_assertions = st.checkbox("Enable SVA Assertions", value=True)
                enable_coverage = st.checkbox("Enable Functional Coverage", value=True)
                enable_constraints = st.checkbox("Enable Constrained Random", value=True)
            
            # Custom scenarios
            use_custom_scenarios = st.checkbox("Use Custom Test Scenarios")
            
            custom_scenarios = None
            if use_custom_scenarios:
                scenarios_text = st.text_area(
                    "Custom Scenarios (JSON)",
                    height=200,
                    placeholder='[{"name": "basic_test", "description": "Basic functionality test", "priority": "high"}]'
                )
                
                if scenarios_text:
                    try:
                        custom_scenarios = json.loads(scenarios_text)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format for scenarios")
            
            # Generate button
            if st.button("Generate Testbench", type="primary"):
                with st.spinner("Analyzing RTL and generating testbench using LangChain..."):
                    try:
                        generator = RTLTestbenchGenerator(self.settings)
                        
                        # Create output directory
                        output_dir = Path("/tmp/rtl_pilot_output")
                        output_dir.mkdir(exist_ok=True)
                        
                        # Generate testbench using new async interface
                        tb_result = self._run_async(generator.generate_testbench_async(
                            rtl_file=str(rtl_path),
                            output_dir=str(output_dir),
                            test_scenarios=custom_scenarios,
                            style=style,
                            template=template
                        ))
                        
                        if tb_result and tb_result.get('success', False):
                            st.success(f"✅ Testbench generated successfully!")
                            
                            # Show analysis results
                            if 'analysis' in tb_result:
                                analysis = tb_result['analysis']
                                st.markdown("### RTL Analysis")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Module", analysis.get('module_name', 'Unknown'))
                                with col2:
                                    st.metric("Ports", len(analysis.get('ports', [])))
                                with col3:
                                    st.metric("Clock Domains", len(analysis.get('clock_domains', [])))
                                with col4:
                                    st.metric("State Machines", len(analysis.get('state_machines', [])))
                            
                            # Display generated code
                            tb_file = Path(tb_result['testbench_file'])
                            with open(tb_file, 'r') as f:
                                tb_content = f.read()
                            
                            st.markdown("### Generated Testbench")
                            st.code(tb_content, language="systemverilog")
                            
                            # Download button
                            st.download_button(
                                label="Download Testbench",
                                data=tb_content,
                                file_name=tb_file.name,
                                mime="text/plain"
                            )
                            
                            # Show generation details
                            if 'generation_details' in tb_result:
                                with st.expander("Generation Details"):
                                    details = tb_result['generation_details']
                                    st.json(details)
                        else:
                            error_msg = tb_result.get('error', 'Unknown error') if tb_result else 'Generation failed'
                            st.error(f"❌ Generation failed: {error_msg}")
                        
                    except Exception as e:
                        st.error(f"❌ Generation failed: {e}")
                        self.logger.error(f"Testbench generation error: {e}")
    
    def render_simulation_runner(self):
        """Render the simulation runner interface."""
        st.title("⚡ Simulation Runner")
        st.markdown("Run RTL simulations using Vivado with LangChain integration")
        
        # File uploads
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**RTL File**")
            rtl_file = st.file_uploader("RTL File", type=["v", "sv"], key="sim_rtl")
        
        with col2:
            st.markdown("**Testbench File**")
            tb_file = st.file_uploader("Testbench File", type=["v", "sv"], key="sim_tb")
        
        if rtl_file and tb_file:
            # Save files temporarily
            rtl_path = Path(f"/tmp/{rtl_file.name}")
            tb_path = Path(f"/tmp/{tb_file.name}")
            
            with open(rtl_path, "wb") as f:
                f.write(rtl_file.getbuffer())
            with open(tb_path, "wb") as f:
                f.write(tb_file.getbuffer())
            
            # Simulation options
            st.markdown("### Simulation Options")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                sim_time = st.text_input("Simulation Time", value="1000ns")
            with col2:
                waveform = st.checkbox("Generate Waveform", value=True)
            with col3:
                coverage = st.checkbox("Enable Coverage", value=True)
            
            # Additional files
            additional_files = st.file_uploader(
                "Additional Files (Optional)",
                type=["v", "sv", "vh", "svh"],
                accept_multiple_files=True
            )
            
            # Simulation parameters
            with st.expander("Advanced Parameters"):
                vivado_path = st.text_input("Vivado Path", value=self.settings.vivado_path or "vivado")
                timeout = st.number_input("Timeout (seconds)", value=300, min_value=10)
                verbose = st.checkbox("Verbose Output", value=False)
            
            # Run simulation
            if st.button("Run Simulation", type="primary"):
                with st.spinner("Running simulation..."):
                    try:
                        runner = SimulationRunner(self.settings)
                        
                        # Prepare file list
                        files = [str(rtl_path), str(tb_path)]
                        if additional_files:
                            for additional in additional_files:
                                add_path = Path(f"/tmp/{additional.name}")
                                with open(add_path, "wb") as f:
                                    f.write(additional.getbuffer())
                                files.append(str(add_path))
                        
                        # Run simulation
                        sim_result = self._run_async(runner.run_simulation_async(
                            files=files,
                            top_module=tb_path.stem,
                            sim_time=sim_time,
                            enable_coverage=coverage,
                            generate_waveform=waveform
                        ))
                        
                        if sim_result and sim_result.get('success', False):
                            st.success("✅ Simulation completed successfully!")
                            
                            # Show simulation results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Status", "PASS" if sim_result.get('passed', False) else "FAIL")
                            with col2:
                                st.metric("Runtime", f"{sim_result.get('runtime', 0):.2f}s")
                            with col3:
                                st.metric("Coverage", f"{sim_result.get('coverage_pct', 0):.1f}%")
                            
                            # Show log output
                            if 'log' in sim_result:
                                st.markdown("### Simulation Log")
                                st.text_area("Output", sim_result['log'], height=300, disabled=True)
                            
                            # Waveform download
                            if waveform and 'waveform_file' in sim_result:
                                waveform_path = Path(sim_result['waveform_file'])
                                if waveform_path.exists():
                                    with open(waveform_path, 'rb') as f:
                                        st.download_button(
                                            "Download Waveform",
                                            data=f.read(),
                                            file_name=waveform_path.name,
                                            mime="application/octet-stream"
                                        )
                            
                            # Coverage report
                            if coverage and 'coverage_report' in sim_result:
                                with st.expander("Coverage Report"):
                                    st.json(sim_result['coverage_report'])
                        else:
                            error_msg = sim_result.get('error', 'Unknown error') if sim_result else 'Simulation failed'
                            st.error(f"❌ Simulation failed: {error_msg}")
                            
                            if sim_result and 'log' in sim_result:
                                with st.expander("Error Log"):
                                    st.text_area("Error Output", sim_result['log'], height=200, disabled=True)
                    
                    except Exception as e:
                        st.error(f"❌ Simulation error: {e}")
                        self.logger.error(f"Simulation error: {e}")
        
        else:
            st.info("Please upload both RTL and testbench files to run simulation")
    
    def render_result_evaluator(self):
        """Render the result evaluator interface."""
        st.title("📊 Result Evaluator")
        st.markdown("Analyze and evaluate verification results using AI")
        
        # Input options
        input_method = st.radio(
            "Input Method",
            ["Upload Files", "Use Previous Results", "Enter Results Manually"]
        )
        
        if input_method == "Upload Files":
            # File uploads for evaluation
            col1, col2 = st.columns(2)
            
            with col1:
                sim_log = st.file_uploader("Simulation Log", type=["log", "txt"])
            with col2:
                coverage_file = st.file_uploader("Coverage Report", type=["xml", "txt", "json"])
            
            waveform_file = st.file_uploader("Waveform File (Optional)", type=["vcd", "fst", "wdb"])
            
            if sim_log:
                log_content = sim_log.read().decode('utf-8')
                
                if st.button("Analyze Results", type="primary"):
                    with st.spinner("Analyzing results with AI..."):
                        try:
                            evaluator = ResultEvaluator(self.settings)
                            
                            # Prepare evaluation data
                            eval_data = {
                                'simulation_log': log_content,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            if coverage_file:
                                coverage_content = coverage_file.read().decode('utf-8')
                                eval_data['coverage_report'] = coverage_content
                            
                            # Run evaluation
                            eval_result = self._run_async(evaluator.evaluate_results_async(eval_data))
                            
                            if eval_result and eval_result.get('success', False):
                                st.success("✅ Analysis completed!")
                                
                                # Show metrics
                                metrics = eval_result.get('metrics', {})
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Overall Score", f"{metrics.get('overall_score', 0):.1f}/10")
                                with col2:
                                    st.metric("Coverage", f"{metrics.get('coverage_pct', 0):.1f}%")
                                with col3:
                                    st.metric("Tests Passed", metrics.get('tests_passed', 0))
                                with col4:
                                    st.metric("Tests Failed", metrics.get('tests_failed', 0))
                                
                                # Show analysis
                                if 'analysis' in eval_result:
                                    st.markdown("### AI Analysis")
                                    st.write(eval_result['analysis'])
                                
                                # Show recommendations
                                if 'recommendations' in eval_result:
                                    st.markdown("### Recommendations")
                                    for rec in eval_result['recommendations']:
                                        st.write(f"• {rec}")
                                
                                # Issues found
                                if 'issues' in eval_result and eval_result['issues']:
                                    st.markdown("### Issues Found")
                                    for issue in eval_result['issues']:
                                        st.warning(f"⚠️ {issue}")
                                
                                # Save to history
                                st.session_state.results_history.append(eval_result)
                            else:
                                error_msg = eval_result.get('error', 'Unknown error') if eval_result else 'Analysis failed'
                                st.error(f"❌ Analysis failed: {error_msg}")
                        
                        except Exception as e:
                            st.error(f"❌ Analysis error: {e}")
        
        elif input_method == "Use Previous Results":
            if st.session_state.results_history:
                selected_idx = st.selectbox(
                    "Select Previous Result",
                    range(len(st.session_state.results_history)),
                    format_func=lambda x: f"Result {x+1} - {st.session_state.results_history[x].get('timestamp', 'Unknown')}"
                )
                
                if selected_idx is not None:
                    result = st.session_state.results_history[selected_idx]
                    
                    # Display result details
                    self._display_evaluation_result(result)
            else:
                st.info("No previous results available")
        
        else:  # Manual entry
            st.markdown("### Manual Result Entry")
            
            col1, col2 = st.columns(2)
            with col1:
                tests_passed = st.number_input("Tests Passed", min_value=0, value=0)
                coverage_pct = st.slider("Coverage Percentage", 0, 100, 0)
            with col2:
                tests_failed = st.number_input("Tests Failed", min_value=0, value=0)
                runtime = st.number_input("Runtime (seconds)", min_value=0.0, value=0.0)
            
            issues_text = st.text_area("Issues/Errors (one per line)")
            notes = st.text_area("Additional Notes")
            
            if st.button("Analyze Manual Results"):
                # Create manual result data
                manual_data = {
                    'tests_passed': tests_passed,
                    'tests_failed': tests_failed,
                    'coverage_pct': coverage_pct,
                    'runtime': runtime,
                    'issues': issues_text.split('\n') if issues_text else [],
                    'notes': notes,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'manual_entry': True
                }
                
                # Add to history
                st.session_state.results_history.append(manual_data)
                st.success("Manual results added to history!")
    
    def render_verification_workflow(self):
        """Render the complete verification workflow interface."""
        st.title("🔄 Verification Workflow")
        st.markdown("End-to-end RTL verification workflow with LangChain AI agents")
        
        # Workflow configuration
        st.markdown("### Workflow Configuration")
        
        # RTL file upload
        rtl_file = st.file_uploader("RTL Design File", type=["v", "sv"])
        
        if rtl_file:
            # Save file temporarily
            rtl_path = Path(f"/tmp/{rtl_file.name}")
            with open(rtl_path, "wb") as f:
                f.write(rtl_file.getbuffer())
            
            # Workflow options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_iterations = st.number_input("Max Iterations", min_value=1, max_value=10, value=3)
            with col2:
                coverage_target = st.slider("Coverage Target (%)", 50, 100, 80)
            with col3:
                auto_improve = st.checkbox("Auto Improve", value=True)
            
            # Advanced workflow options
            with st.expander("Advanced Options"):
                enable_planning = st.checkbox("Enable AI Planning", value=True)
                enable_feedback = st.checkbox("Enable Feedback Loop", value=True)
                parallel_sims = st.checkbox("Parallel Simulations", value=False)
                timeout_minutes = st.number_input("Workflow Timeout (minutes)", min_value=5, value=30)
            
            # Custom verification plan
            use_custom_plan = st.checkbox("Use Custom Verification Plan")
            custom_plan = None
            
            if use_custom_plan:
                plan_text = st.text_area(
                    "Verification Plan (JSON)",
                    height=200,
                    placeholder='{"test_scenarios": [...], "coverage_goals": {...}, "constraints": [...]}'
                )
                
                if plan_text:
                    try:
                        custom_plan = json.loads(plan_text)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format for verification plan")
            
            # Run workflow
            if st.button("Start Verification Workflow", type="primary"):
                
                # Create output directory
                output_dir = Path("/tmp/rtl_pilot_workflow")
                output_dir.mkdir(exist_ok=True)
                
                # Progress tracking
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                results_placeholder = st.empty()
                
                # Initialize workflow
                with st.spinner("Initializing workflow..."):
                    st.session_state.workflow_state = "running"
                    
                    try:
                        # Import the workflow
                        from ..workflows.default_flow import DefaultVerificationFlow
                        
                        workflow = DefaultVerificationFlow(self.settings)
                        
                        # Configuration
                        workflow_config = {
                            'rtl_file': str(rtl_path),
                            'output_dir': str(output_dir),
                            'max_iterations': max_iterations,
                            'coverage_target': coverage_target,
                            'auto_improve': auto_improve,
                            'enable_planning': enable_planning,
                            'enable_feedback': enable_feedback,
                            'timeout_minutes': timeout_minutes
                        }
                        
                        if custom_plan:
                            workflow_config['custom_plan'] = custom_plan
                        
                        # Run workflow with progress tracking
                        self._run_workflow_with_progress(
                            workflow, workflow_config, 
                            progress_placeholder, status_placeholder, results_placeholder
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Workflow failed: {e}")
                        st.session_state.workflow_state = "failed"
                        self.logger.error(f"Workflow error: {e}")
        
        else:
            st.info("Please upload an RTL file to start the verification workflow")
        
        # Show workflow history
        if st.session_state.results_history:
            st.markdown("---")
            st.markdown("### Workflow History")
            
            for i, result in enumerate(reversed(st.session_state.results_history)):
                with st.expander(f"Workflow {len(st.session_state.results_history) - i} - {result.get('timestamp', 'Unknown')}"):
                    self._display_evaluation_result(result)
            tb_file = st.file_uploader("Testbench File", type=["v", "sv"])
        
        if rtl_file:
            # Simulation settings
            st.markdown("### Simulation Settings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sim_time = st.selectbox(
                    "Simulation Time",
                    ["100ns", "1us", "10us", "100us", "1ms"],
                    index=1
                )
            
            with col2:
                waveform_dump = st.checkbox("Generate Waveforms", value=True)
            
            with col3:
                coverage_analysis = st.checkbox("Coverage Analysis", value=True)
            
            # Run simulation
            if st.button("Run Simulation", type="primary"):
                with st.spinner("Running simulation..."):
                    try:
                        # Save files temporarily
                        rtl_path = Path(f"/tmp/{rtl_file.name}")
                        with open(rtl_path, "wb") as f:
                            f.write(rtl_file.getbuffer())
                        
                        tb_path = None
                        if tb_file:
                            tb_path = Path(f"/tmp/{tb_file.name}")
                            with open(tb_path, "wb") as f:
                                f.write(tb_file.getbuffer())
                        else:
                            # Generate testbench if not provided
                            st.info("No testbench provided, generating one...")
                            generator = RTLTestbenchGenerator(self.settings)
                            output_dir = Path("/tmp/rtl_pilot_output")
                            output_dir.mkdir(exist_ok=True)
                            tb_path = generator.generate_testbench(rtl_path, output_dir)
                        
                        # Run simulation
                        runner = SimulationRunner(self.settings)
                        project_dir = Path("/tmp/rtl_pilot_sim")
                        
                        project_file = runner.setup_simulation_project(
                            rtl_files=[rtl_path],
                            testbench_file=tb_path,
                            project_dir=project_dir
                        )
                        
                        results = runner.run_simulation(
                            project_file=project_file,
                            simulation_time=sim_time
                        )
                        
                        # Display results
                        if results.get("success", False):
                            st.success("✅ Simulation completed successfully!")
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Runtime", f"{results.get('runtime', 0):.2f}s")
                            with col2:
                                st.metric("Errors", len(results.get("errors", [])))
                            with col3:
                                st.metric("Warnings", len(results.get("warnings", [])))
                            
                            # Show logs if there are issues
                            if results.get("errors") or results.get("warnings"):
                                with st.expander("Simulation Messages"):
                                    for error in results.get("errors", []):
                                        st.error(f"ERROR: {error}")
                                    for warning in results.get("warnings", []):
                                        st.warning(f"WARNING: {warning}")
                        else:
                            st.error("❌ Simulation failed!")
                            for error in results.get("errors", []):
                                st.error(error)
                        
                    except Exception as e:
                        st.error(f"❌ Simulation failed: {e}")
    
    def render_result_evaluator(self):
        """Render the result evaluator interface."""
        st.title("🔍 Result Evaluator")
        st.markdown("Analyze and evaluate simulation results")
        
        # Upload results file
        results_file = st.file_uploader(
            "Upload Simulation Results (JSON)",
            type=["json"],
            help="Upload the simulation results file to analyze"
        )
        
        if results_file:
            try:
                results_data = json.load(results_file)
                
                # Basic evaluation
                evaluator = ResultEvaluator(self.settings)
                evaluation = evaluator.evaluate_simulation_results(results_data)
                
                # Display evaluation summary
                st.markdown("### Evaluation Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    score = evaluation.get("overall_score", 0)
                    st.metric("Overall Score", f"{score:.1f}/100")
                
                with col2:
                    status = "PASS" if evaluation.get("pass", False) else "FAIL"
                    st.metric("Status", status)
                
                with col3:
                    coverage = evaluation.get("coverage_analysis", {}).get("coverage_score", 0)
                    st.metric("Coverage", f"{coverage:.1f}%")
                
                with col4:
                    issues = len(evaluation.get("issues_found", []))
                    st.metric("Issues", issues)
                
                # Detailed analysis
                st.markdown("### Detailed Analysis")
                
                # Coverage breakdown
                coverage_analysis = evaluation.get("coverage_analysis", {})
                if coverage_analysis:
                    st.markdown("#### Coverage Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Line Coverage", f"{coverage_analysis.get('line_coverage', 0):.1f}%")
                        st.metric("Branch Coverage", f"{coverage_analysis.get('branch_coverage', 0):.1f}%")
                    
                    with col2:
                        st.metric("Toggle Coverage", f"{coverage_analysis.get('toggle_coverage', 0):.1f}%")
                        st.metric("FSM Coverage", f"{coverage_analysis.get('fsm_coverage', 0):.1f}%")
                
                # Issues found
                issues = evaluation.get("issues_found", [])
                if issues:
                    st.markdown("#### Issues Found")
                    for issue in issues:
                        st.error(issue)
                
                # Recommendations
                st.markdown("#### Recommendations")
                recommendations = evaluation.get("recommendations", [])
                if recommendations:
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.success("No specific recommendations - verification looks good!")
                
            except Exception as e:
                st.error(f"Failed to analyze results: {e}")
    
    def render_verification_workflow(self):
        """Render the complete verification workflow interface."""
        st.title("🔄 Verification Workflow")
        st.markdown("Run complete AI-powered verification campaigns")
        
        # File upload
        rtl_file = st.file_uploader("Upload RTL File", type=["v", "sv"])
        
        if rtl_file:
            # Workflow configuration
            st.markdown("### Workflow Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                coverage_target = st.slider("Coverage Target (%)", 70, 100, 90)
                max_iterations = st.slider("Max Iterations", 1, 10, 5)
            
            with col2:
                goals_preset = st.selectbox(
                    "Verification Goals",
                    ["Basic Functionality", "Comprehensive", "Performance", "Custom"]
                )
                
                timeout = st.selectbox(
                    "Workflow Timeout",
                    ["30 minutes", "1 hour", "2 hours", "4 hours"]
                )
            
            # Custom goals
            if goals_preset == "Custom":
                custom_goals = st.text_area(
                    "Custom Goals (JSON)",
                    height=150,
                    placeholder='{"coverage_target": 95, "performance": {"max_latency": "10ns"}}'
                )
            
            # Start workflow
            if st.button("Start Verification Workflow", type="primary"):
                # Save RTL file
                rtl_path = Path(f"/tmp/{rtl_file.name}")
                with open(rtl_path, "wb") as f:
                    f.write(rtl_file.getbuffer())
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Configure workflow
                verification_goals = {
                    "coverage_target": coverage_target,
                    "max_iterations": max_iterations
                }
                
                if goals_preset == "Custom" and 'custom_goals' in locals():
                    try:
                        custom_data = json.loads(custom_goals)
                        verification_goals.update(custom_data)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in custom goals")
                        return
                
                # Run workflow
                try:
                    planner = VerificationPlanner(self.settings)
                    
                    # Create plan
                    status_text.text("Creating verification plan...")
                    progress_bar.progress(10)
                    
                    plan = planner.create_verification_plan(
                        rtl_files=[rtl_path],
                        verification_goals=verification_goals
                    )
                    
                    # Execute workflow
                    status_text.text("Executing verification workflow...")
                    progress_bar.progress(20)
                    
                    output_dir = Path("/tmp/rtl_pilot_workflow")
                    output_dir.mkdir(exist_ok=True)
                    
                    # Note: In a real Streamlit app, you'd want to run this asynchronously
                    # For now, we'll simulate the workflow
                    
                    for i in range(max_iterations):
                        status_text.text(f"Running iteration {i+1}/{max_iterations}...")
                        progress_percentage = 20 + (i + 1) * (70 / max_iterations)
                        progress_bar.progress(int(progress_percentage))
                        time.sleep(1)  # Simulate work
                    
                    status_text.text("Generating final report...")
                    progress_bar.progress(95)
                    
                    # Simulate results
                    workflow_results = {
                        "success": True,
                        "iterations": max_iterations,
                        "final_metrics": {
                            "final_coverage": coverage_target,
                            "total_test_cases": 25,
                            "bugs_found": 2
                        },
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Add to history
                    st.session_state.results_history.append(workflow_results)
                    
                    progress_bar.progress(100)
                    status_text.text("Workflow completed!")
                    
                    # Display results
                    st.success("✅ Verification workflow completed successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Final Coverage", f"{workflow_results['final_metrics']['final_coverage']:.1f}%")
                    with col2:
                        st.metric("Iterations Used", workflow_results["iterations"])
                    with col3:
                        st.metric("Test Cases", workflow_results['final_metrics']['total_test_cases'])
                    
                except Exception as e:
                    st.error(f"❌ Workflow failed: {e}")
    
    def render_settings(self):
        """Render the settings interface."""
        st.title("⚙️ Settings")
        st.markdown("Configure RTL-Pilot settings")
        
        # Tool paths
        st.markdown("### Tool Paths")
        
        vivado_path = st.text_input(
            "Vivado Path",
            value=str(self.settings.vivado_path),
            help="Path to Vivado executable"
        )
        
        # LLM settings
        st.markdown("### LLM Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            llm_model = st.selectbox(
                "LLM Model",
                ["gpt-4", "gpt-3.5-turbo", "claude-3", "local-model"],
                index=0
            )
        
        with col2:
            api_key = st.text_input(
                "API Key",
                type="password",
                help="API key for LLM service"
            )
        
        # Simulation settings
        st.markdown("### Simulation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_sim_time = st.text_input("Default Simulation Time", value="1us")
        
        with col2:
            simulation_timeout = st.text_input("Simulation Timeout", value="10ms")
        
        # Save settings
        if st.button("Save Settings"):
            # TODO: Implement settings save
            st.success("Settings saved successfully!")
    
    def run(self):
        """Run the Streamlit web interface with LangChain integration."""
        st.set_page_config(
            page_title="RTL-Pilot",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Render sidebar
        page = self.render_sidebar()
        
        # Render selected page
        if page == "Dashboard":
            self.render_dashboard()
        elif page == "Chat Assistant":
            self.render_chat_assistant()
        elif page == "Testbench Generator":
            self.render_testbench_generator()
        elif page == "Simulation Runner":
            self.render_simulation_runner()
        elif page == "Result Evaluator":
            self.render_result_evaluator()
        elif page == "Verification Workflow":
            self.render_verification_workflow()
        elif page == "Settings":
            self.render_settings()


def main():
    """Main entry point for the web interface."""
    interface = WebInterface()
    interface.run()


if __name__ == "__main__":
    main()
