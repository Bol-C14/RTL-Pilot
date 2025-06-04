"""
测试结果评估器模块
"""
import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from rtl_pilot.agents.evaluation import ResultEvaluator
from rtl_pilot.config.settings import Settings


@pytest.fixture
def settings():
    """测试设置fixture"""
    return Settings(
        llm_provider="openai",
        llm_model="gpt-4",
        workspace_path=tempfile.mkdtemp()
    )


@pytest.fixture
def evaluator(settings):
    """结果评估器fixture"""
    return ResultEvaluator(settings)


@pytest.fixture
def sample_simulation_results():
    """示例仿真结果fixture"""
    return {
        'success': True,
        'log_file': 'simulation.log',
        'log_content': '''
Info: Starting simulation
Info: Test case 1 passed
Warning: Signal x is undefined
Error: Assertion failed at time 100ns
Info: Test case 2 passed
Info: Simulation completed
Coverage: 85.5%
        ''',
        'waveform_file': 'dump.vcd',
        'coverage_file': 'coverage.xml',
        'execution_time': 125.5
    }


@pytest.fixture
def sample_coverage_xml():
    """示例覆盖率XML fixture"""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<coverage>
    <file name="adder.v">
        <line_coverage>
            <line number="10" hits="5"/>
            <line number="11" hits="0"/>
            <line number="12" hits="3"/>
        </line_coverage>
        <branch_coverage>
            <branch id="1" taken="true"/>
            <branch id="2" taken="false"/>
        </branch_coverage>
    </file>
    <summary>
        <line_rate>0.855</line_rate>
        <branch_rate>0.750</branch_rate>
        <overall_rate>0.802</overall_rate>
    </summary>
</coverage>'''


class TestResultEvaluator:
    """结果评估器测试类"""
    
    def test_init(self, evaluator, settings):
        """测试初始化"""
        assert evaluator.settings == settings
        assert hasattr(evaluator, 'llm_client')
    
    @pytest.mark.asyncio
    async def test_evaluate_results_success(self, evaluator, sample_simulation_results):
        """测试成功的结果评估"""
        with patch.object(evaluator, '_parse_log_content') as mock_parse_log, \
             patch.object(evaluator, '_analyze_coverage') as mock_analyze_cov, \
             patch.object(evaluator, '_generate_feedback') as mock_feedback:
            
            # 设置mock返回值
            mock_parse_log.return_value = {
                'passed_tests': 2,
                'failed_tests': 1,
                'warnings': 1,
                'errors': 1
            }
            mock_analyze_cov.return_value = {
                'line_coverage': 85.5,
                'branch_coverage': 75.0,
                'overall_coverage': 80.2
            }
            mock_feedback.return_value = {
                'summary': 'Good test coverage but some failures detected',
                'recommendations': ['Fix assertion error', 'Improve coverage']
            }
            
            result = await evaluator.evaluate_results(sample_simulation_results)
            
            assert result['success'] is True
            assert 'log_analysis' in result
            assert 'coverage_analysis' in result
            assert 'feedback' in result
    
    def test_parse_log_content(self, evaluator):
        """测试日志内容解析"""
        log_content = '''
Info: Starting simulation
Info: Test case 1 passed
Warning: Signal x is undefined
Error: Assertion failed at time 100ns
Info: Test case 2 passed
Info: Simulation completed
        '''
        
        analysis = evaluator._parse_log_content(log_content)
        
        assert analysis['passed_tests'] == 2
        assert analysis['failed_tests'] == 0  # 错误不等于失败的测试
        assert analysis['warnings'] == 1
        assert analysis['errors'] == 1
        assert analysis['info_messages'] == 4
    
    def test_analyze_coverage(self, evaluator, sample_coverage_xml):
        """测试覆盖率分析"""
        with patch('builtins.open', mock_open(read_data=sample_coverage_xml)):
            coverage = evaluator._analyze_coverage('coverage.xml')
            
            assert coverage['line_coverage'] == 85.5
            assert coverage['branch_coverage'] == 75.0
            assert coverage['overall_coverage'] == 80.2
            assert 'file_details' in coverage
    
    def test_analyze_coverage_file_not_found(self, evaluator):
        """测试覆盖率文件不存在的处理"""
        coverage = evaluator._analyze_coverage('nonexistent.xml')
        
        assert coverage['line_coverage'] == 0
        assert coverage['branch_coverage'] == 0
        assert coverage['overall_coverage'] == 0
        assert 'error' in coverage
    
    @pytest.mark.asyncio
    async def test_generate_feedback(self, evaluator):
        """测试反馈生成"""
        log_analysis = {
            'passed_tests': 5,
            'failed_tests': 1,
            'warnings': 2,
            'errors': 1
        }
        coverage_analysis = {
            'line_coverage': 75.0,
            'branch_coverage': 60.0,
            'overall_coverage': 67.5
        }
        
        with patch.object(evaluator.llm_client, 'chat') as mock_chat:
            mock_chat.return_value = Mock(
                choices=[Mock(
                    message=Mock(content='''
{
    "summary": "Mixed results with room for improvement",
    "issues": ["One test failure", "Low branch coverage"],
    "recommendations": ["Fix failing test", "Add edge case tests"]
}
                    ''')
                )]
            )
            
            feedback = await evaluator._generate_feedback(log_analysis, coverage_analysis)
            
            assert 'summary' in feedback
            assert 'issues' in feedback
            assert 'recommendations' in feedback
    
    def test_extract_coverage_metrics(self, evaluator):
        """测试覆盖率指标提取"""
        log_content = '''
        Line Coverage: 85.5%
        Branch Coverage: 75.0%
        Toggle Coverage: 92.3%
        Overall Coverage: 84.2%
        '''
        
        metrics = evaluator._extract_coverage_metrics(log_content)
        
        assert metrics['line_coverage'] == 85.5
        assert metrics['branch_coverage'] == 75.0
        assert metrics['toggle_coverage'] == 92.3
        assert metrics['overall_coverage'] == 84.2
    
    def test_classify_log_severity(self, evaluator):
        """测试日志严重程度分类"""
        messages = [
            "Info: Test passed",
            "Warning: Signal not driven",
            "Error: Syntax error",
            "Fatal: Cannot continue",
            "Note: Informational message"
        ]
        
        for message in messages:
            severity = evaluator._classify_log_severity(message)
            if "Info" in message or "Note" in message:
                assert severity == "info"
            elif "Warning" in message:
                assert severity == "warning"
            elif "Error" in message:
                assert severity == "error"
            elif "Fatal" in message:
                assert severity == "fatal"
    
    def test_parse_waveform_info(self, evaluator):
        """测试波形信息解析"""
        with tempfile.NamedTemporaryFile(suffix='.vcd', delete=False) as temp_file:
            temp_file.write(b'$version VCD $end\n$timescale 1ns $end\n')
            temp_file.flush()
            
            waveform_info = evaluator._parse_waveform_info(temp_file.name)
            
            assert waveform_info['file_exists'] is True
            assert waveform_info['file_size'] > 0
            assert 'timescale' in waveform_info
    
    @pytest.mark.asyncio
    async def test_evaluate_results_with_no_coverage(self, evaluator):
        """测试无覆盖率文件的结果评估"""
        results = {
            'success': True,
            'log_content': 'Simple simulation log',
            'coverage_file': None
        }
        
        with patch.object(evaluator, '_generate_feedback') as mock_feedback:
            mock_feedback.return_value = {'summary': 'No coverage data'}
            
            evaluation = await evaluator.evaluate_results(results)
            
            assert evaluation['success'] is True
            assert evaluation['coverage_analysis']['overall_coverage'] == 0
    
    def test_performance_analysis(self, evaluator, sample_simulation_results):
        """测试性能分析"""
        analysis = evaluator._analyze_performance(sample_simulation_results)
        
        assert analysis['execution_time'] == 125.5
        assert 'performance_rating' in analysis
        assert analysis['performance_rating'] in ['excellent', 'good', 'poor']
    
    def test_error_pattern_detection(self, evaluator):
        """测试错误模式检测"""
        log_content = '''
Error: Division by zero at line 45
Error: Signal 'clk' not found
Warning: Latch inferred for signal 'data'
Error: Division by zero at line 67
        '''
        
        patterns = evaluator._detect_error_patterns(log_content)
        
        assert 'Division by zero' in patterns
        assert patterns['Division by zero'] == 2
        assert 'Signal not found' in patterns
        assert 'Latch inferred' in patterns
    
    @pytest.mark.asyncio
    async def test_generate_improvement_suggestions(self, evaluator):
        """测试改进建议生成"""
        evaluation_data = {
            'log_analysis': {'errors': 2, 'warnings': 3},
            'coverage_analysis': {'overall_coverage': 65.0},
            'performance_analysis': {'execution_time': 200.0}
        }
        
        with patch.object(evaluator.llm_client, 'chat') as mock_chat:
            mock_chat.return_value = Mock(
                choices=[Mock(
                    message=Mock(content='''
{
    "testbench_improvements": ["Add more edge cases", "Improve stimulus"],
    "rtl_insights": ["Optimize critical path", "Reduce logic depth"],
    "coverage_goals": ["Target 90% line coverage", "Improve branch coverage"]
}
                    ''')
                )]
            )
            
            suggestions = await evaluator._generate_improvement_suggestions(evaluation_data)
            
            assert 'testbench_improvements' in suggestions
            assert 'rtl_insights' in suggestions
            assert 'coverage_goals' in suggestions


class TestCoverageAnalysis:
    """覆盖率分析测试类"""
    
    def test_parse_xml_coverage(self, evaluator, sample_coverage_xml):
        """测试XML覆盖率解析"""
        root = ET.fromstring(sample_coverage_xml)
        coverage_data = evaluator._parse_xml_coverage(root)
        
        assert 'files' in coverage_data
        assert 'summary' in coverage_data
        assert coverage_data['summary']['line_rate'] == 0.855
    
    def test_calculate_coverage_score(self, evaluator):
        """测试覆盖率评分计算"""
        coverage_data = {
            'line_coverage': 85.0,
            'branch_coverage': 75.0,
            'toggle_coverage': 90.0
        }
        
        score = evaluator._calculate_coverage_score(coverage_data)
        
        assert 0 <= score <= 100
        assert isinstance(score, float)
    
    def test_identify_uncovered_lines(self, evaluator, sample_coverage_xml):
        """测试未覆盖行识别"""
        with patch('builtins.open', mock_open(read_data=sample_coverage_xml)):
            uncovered = evaluator._identify_uncovered_lines('coverage.xml')
            
            assert 'adder.v' in uncovered
            assert 11 in uncovered['adder.v']  # line with 0 hits


if __name__ == '__main__':
    pytest.main([__file__])
