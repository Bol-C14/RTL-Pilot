"""
测试验证规划器模块
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import json

from rtl_pilot.agents.planner import VerificationPlanner
from rtl_pilot.config.settings import Settings


@pytest.fixture
def settings():
    """测试设置fixture"""
    return Settings(
        llm_provider="openai",
        llm_model="gpt-4",
        workspace_path=tempfile.mkdtemp(),
        max_iterations=3
    )


@pytest.fixture
def planner(settings):
    """验证规划器fixture"""
    return VerificationPlanner(settings)


@pytest.fixture
def sample_project_config():
    """示例项目配置fixture"""
    return {
        'project_name': 'test_project',
        'rtl_files': ['adder.v', 'utils.v'],
        'top_module': 'adder',
        'verification_goals': {
            'line_coverage_target': 90,
            'branch_coverage_target': 85,
            'functional_tests': ['basic_ops', 'edge_cases', 'stress_test']
        },
        'constraints': {
            'max_simulation_time': '1000ns',
            'timeout': 300
        }
    }


@pytest.fixture
def sample_rtl_analysis():
    """示例RTL分析结果fixture"""
    return {
        'modules': ['adder', 'utils'],
        'interfaces': {
            'adder': {
                'inputs': ['a[3:0]', 'b[3:0]', 'cin'],
                'outputs': ['sum[3:0]', 'cout']
            }
        },
        'complexity_score': 3.5,
        'estimated_test_scenarios': 15
    }


class TestVerificationPlanner:
    """验证规划器测试类"""
    
    def test_init(self, planner, settings):
        """测试初始化"""
        assert planner.settings == settings
        assert hasattr(planner, 'llm_client')
        assert hasattr(planner, 'tb_generator')
        assert hasattr(planner, 'sim_runner')
        assert hasattr(planner, 'evaluator')
    
    @pytest.mark.asyncio
    async def test_create_verification_plan(self, planner, sample_project_config):
        """测试验证计划创建"""
        with patch.object(planner, '_analyze_rtl_complexity') as mock_analyze, \
             patch.object(planner, '_generate_test_strategy') as mock_strategy:
            
            mock_analyze.return_value = {
                'complexity': 'medium',
                'estimated_effort': 'moderate'
            }
            mock_strategy.return_value = {
                'phases': ['basic', 'corner_cases', 'stress'],
                'priority_tests': ['functional', 'boundary']
            }
            
            plan = await planner.create_verification_plan(sample_project_config)
            
            assert 'strategy' in plan
            assert 'phases' in plan
            assert 'estimated_effort' in plan
            mock_analyze.assert_called_once()
            mock_strategy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_plan_success(self, planner, sample_project_config):
        """测试计划执行成功"""
        verification_plan = {
            'phases': [
                {'name': 'basic', 'tests': ['test1', 'test2']},
                {'name': 'advanced', 'tests': ['test3']}
            ],
            'strategy': {'approach': 'incremental'}
        }
        
        with patch.object(planner, '_execute_phase') as mock_execute:
            mock_execute.return_value = {
                'success': True,
                'coverage': 85.0,
                'tests_passed': 2
            }
            
            result = await planner.execute_plan(verification_plan, sample_project_config)
            
            assert result['success'] is True
            assert result['overall_coverage'] > 0
            assert mock_execute.call_count == 2  # 两个阶段
    
    @pytest.mark.asyncio
    async def test_execute_phase(self, planner, sample_project_config):
        """测试阶段执行"""
        phase = {
            'name': 'basic_tests',
            'tests': ['add_test', 'carry_test'],
            'coverage_target': 80
        }
        
        with patch.object(planner.tb_generator, 'generate_testbench') as mock_gen_tb, \
             patch.object(planner.sim_runner, 'run_simulation') as mock_sim, \
             patch.object(planner.evaluator, 'evaluate_results') as mock_eval:
            
            mock_gen_tb.return_value = {'testbench': 'generated_tb.sv'}
            mock_sim.return_value = {'success': True, 'log_file': 'sim.log'}
            mock_eval.return_value = {
                'success': True,
                'coverage_analysis': {'overall_coverage': 82.0}
            }
            
            result = await planner._execute_phase(phase, sample_project_config)
            
            assert result['success'] is True
            assert result['coverage'] == 82.0
            mock_gen_tb.assert_called_once()
            mock_sim.assert_called_once()
            mock_eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_adaptive_planning(self, planner, sample_project_config):
        """测试自适应规划"""
        initial_results = {
            'coverage_analysis': {'overall_coverage': 60.0},
            'log_analysis': {'failed_tests': 2}
        }
        
        with patch.object(planner.llm_client, 'chat') as mock_chat:
            mock_chat.return_value = Mock(
                choices=[Mock(
                    message=Mock(content='''
{
    "revised_strategy": "focus_on_edge_cases",
    "additional_tests": ["boundary_test", "overflow_test"],
    "priority_adjustments": ["increase_stimulus_coverage"]
}
                    ''')
                )]
            )
            
            revised_plan = await planner._adapt_plan_based_on_results(
                sample_project_config, initial_results
            )
            
            assert 'revised_strategy' in revised_plan
            assert 'additional_tests' in revised_plan
    
    def test_analyze_rtl_complexity(self, planner, sample_rtl_analysis):
        """测试RTL复杂度分析"""
        with patch.object(planner, '_parse_rtl_files') as mock_parse:
            mock_parse.return_value = sample_rtl_analysis
            
            complexity = planner._analyze_rtl_complexity(['adder.v'])
            
            assert 'complexity_score' in complexity
            assert 'modules_count' in complexity
            assert 'estimated_scenarios' in complexity
    
    @pytest.mark.asyncio
    async def test_generate_test_strategy(self, planner, sample_project_config):
        """测试测试策略生成"""
        rtl_analysis = {
            'complexity_score': 4.0,
            'interfaces': {'inputs': 3, 'outputs': 2}
        }
        
        with patch.object(planner.llm_client, 'chat') as mock_chat:
            mock_chat.return_value = Mock(
                choices=[Mock(
                    message=Mock(content='''
{
    "approach": "layered_testing",
    "phases": [
        {"name": "basic", "priority": 1},
        {"name": "corner_cases", "priority": 2}
    ],
    "test_categories": ["functional", "boundary", "stress"]
}
                    ''')
                )]
            )
            
            strategy = await planner._generate_test_strategy(
                sample_project_config, rtl_analysis
            )
            
            assert 'approach' in strategy
            assert 'phases' in strategy
            assert 'test_categories' in strategy
    
    def test_estimate_verification_effort(self, planner):
        """测试验证工作量估算"""
        project_metrics = {
            'rtl_lines': 500,
            'modules_count': 3,
            'complexity_score': 3.5,
            'interface_count': 15
        }
        
        effort = planner._estimate_verification_effort(project_metrics)
        
        assert 'estimated_hours' in effort
        assert 'confidence_level' in effort
        assert 'complexity_rating' in effort
    
    @pytest.mark.asyncio
    async def test_optimization_loop(self, planner, sample_project_config):
        """测试优化循环"""
        current_results = {
            'coverage_analysis': {'overall_coverage': 75.0},
            'log_analysis': {'errors': 1, 'warnings': 3}
        }
        
        with patch.object(planner, '_identify_coverage_gaps') as mock_gaps, \
             patch.object(planner, '_generate_targeted_tests') as mock_tests:
            
            mock_gaps.return_value = ['uncovered_branch_1', 'uncovered_line_45']
            mock_tests.return_value = ['gap_test_1.sv', 'gap_test_2.sv']
            
            optimization = await planner._run_optimization_loop(
                current_results, sample_project_config
            )
            
            assert 'targeted_tests' in optimization
            assert 'improvement_strategy' in optimization
    
    def test_progress_tracking(self, planner):
        """测试进度跟踪"""
        phases_results = [
            {'coverage': 30.0, 'tests_passed': 5},
            {'coverage': 65.0, 'tests_passed': 12},
            {'coverage': 85.0, 'tests_passed': 18}
        ]
        
        progress = planner._track_progress(phases_results)
        
        assert progress['current_coverage'] == 85.0
        assert progress['total_tests'] == 18
        assert progress['coverage_trend'] == 'improving'
    
    @pytest.mark.asyncio
    async def test_resource_optimization(self, planner):
        """测试资源优化"""
        resource_usage = {
            'simulation_time': 45.0,
            'memory_usage': '2GB',
            'cpu_utilization': 85.0
        }
        
        optimization = await planner._optimize_resources(resource_usage)
        
        assert 'recommendations' in optimization
        assert 'efficiency_score' in optimization
    
    def test_risk_assessment(self, planner, sample_project_config):
        """测试风险评估"""
        project_data = {
            'complexity': 'high',
            'timeline': 'tight',
            'resources': 'limited'
        }
        
        risks = planner._assess_risks(sample_project_config, project_data)
        
        assert 'risk_level' in risks
        assert 'risk_factors' in risks
        assert 'mitigation_strategies' in risks
    
    @pytest.mark.asyncio
    async def test_plan_validation(self, planner):
        """测试计划验证"""
        verification_plan = {
            'phases': [{'name': 'test', 'tests': ['t1']}],
            'timeline': '2 weeks',
            'resources': {'engineers': 2}
        }
        
        validation = await planner._validate_plan(verification_plan)
        
        assert 'is_valid' in validation
        assert 'issues' in validation
        assert 'recommendations' in validation
    
    def test_metrics_collection(self, planner):
        """测试指标收集"""
        execution_data = {
            'start_time': '2024-01-01T10:00:00',
            'end_time': '2024-01-01T12:30:00',
            'phases_completed': 3,
            'total_tests': 25
        }
        
        metrics = planner._collect_metrics(execution_data)
        
        assert 'execution_duration' in metrics
        assert 'test_velocity' in metrics
        assert 'efficiency_rating' in metrics


class TestPlanOptimization:
    """计划优化测试类"""
    
    @pytest.mark.asyncio
    async def test_coverage_driven_optimization(self, planner):
        """测试基于覆盖率的优化"""
        coverage_data = {
            'line_coverage': 75.0,
            'branch_coverage': 60.0,
            'uncovered_lines': [45, 67, 89]
        }
        
        with patch.object(planner.llm_client, 'chat') as mock_chat:
            mock_chat.return_value = Mock(
                choices=[Mock(
                    message=Mock(content='''
{
    "optimization_strategy": "targeted_coverage",
    "priority_areas": ["branch_coverage", "line_45"],
    "suggested_tests": ["edge_case_test", "boundary_test"]
}
                    ''')
                )]
            )
            
            optimization = await planner._optimize_for_coverage(coverage_data)
            
            assert 'optimization_strategy' in optimization
            assert 'priority_areas' in optimization
    
    def test_test_prioritization(self, planner):
        """测试测试优先级排序"""
        test_candidates = [
            {'name': 'basic_test', 'coverage_impact': 30, 'effort': 2},
            {'name': 'edge_test', 'coverage_impact': 45, 'effort': 5},
            {'name': 'stress_test', 'coverage_impact': 25, 'effort': 8}
        ]
        
        prioritized = planner._prioritize_tests(test_candidates)
        
        assert len(prioritized) == 3
        assert prioritized[0]['name'] == 'edge_test'  # 最高优先级
    
    def test_resource_allocation(self, planner):
        """测试资源分配"""
        available_resources = {
            'engineers': 2,
            'compute_hours': 100,
            'timeline_days': 10
        }
        
        plan_requirements = {
            'estimated_effort': 80,
            'phases': 4,
            'complexity': 'medium'
        }
        
        allocation = planner._allocate_resources(available_resources, plan_requirements)
        
        assert 'phase_allocation' in allocation
        assert 'resource_utilization' in allocation
        assert allocation['feasible'] in [True, False]


if __name__ == '__main__':
    pytest.main([__file__])
