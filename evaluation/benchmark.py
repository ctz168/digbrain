#!/usr/bin/env python3
"""
DigBrain 基准测试和评分模块
Comprehensive Benchmark and Scoring System

功能：
1. 数学能力测试（算术、代数、几何）
2. 代码能力测试（Python编程、算法）
3. 知识问答测试（百科知识、常识推理）
4. 逻辑推理测试（演绎推理、归纳推理）
5. 创造性写作测试

评分机制：
- 每个维度100分制
- 标准答案对比
- 准确率和F1分数计算
"""

import os
import sys
import json
import time
import math
import psutil
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator, Callable, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

OUTPUT_PATH = os.path.join(BASE_DIR, "evaluation/results")
DATASETS_PATH = os.path.join(BASE_DIR, "evaluation/datasets")


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class TestQuestion:
    """测试问题"""
    id: str
    question: str
    category: str  # 类别：arithmetic, algebra, geometry, python, algorithm, etc.
    difficulty: str  # 难度：easy, medium, hard
    expected_answers: List[str]  # 可接受的答案列表
    keywords: List[str] = field(default_factory=list)  # 代码题关键词
    scoring_rubric: Dict[str, float] = field(default_factory=dict)  # 评分细则
    max_score: float = 100.0
    time_limit: float = 30.0  # 时间限制（秒）
    metadata: Dict = field(default_factory=dict)


@dataclass
class AnswerResult:
    """答题结果"""
    question_id: str
    question: str
    expected: List[str]
    response: str
    is_correct: bool
    score: float
    max_score: float
    time_taken: float
    match_type: str  # exact, partial, keyword, semantic
    details: Dict = field(default_factory=dict)


@dataclass
class DimensionResult:
    """维度评估结果"""
    dimension: str
    dimension_name: str
    total_score: float
    max_score: float
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    avg_time: float
    questions: List[AnswerResult] = field(default_factory=list)
    subcategories: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    tokens_per_second: float
    time_to_first_token: float
    total_inference_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    latency_ms: float
    throughput: float
    stdp_updates: int = 0
    memory_operations: int = 0


@dataclass
class BenchmarkReport:
    """完整基准测试报告"""
    timestamp: str
    model_name: str
    total_time: float
    overall_score: float
    dimensions: Dict[str, DimensionResult]
    performance: PerformanceMetrics
    comparison: Dict[str, Dict] = field(default_factory=dict)  # 与基准对比
    history_comparison: List[Dict] = field(default_factory=list)  # 历史对比


# ============================================================================
# 测试数据集
# ============================================================================

class TestDatasets:
    """测试数据集管理器"""
    
    def __init__(self):
        self.datasets: Dict[str, List[TestQuestion]] = {}
        self._load_builtin_datasets()
        self._load_external_datasets()
    
    def _load_builtin_datasets(self):
        """加载内置数据集"""
        
        # ==================== 数学能力测试 ====================
        self.datasets["math"] = [
            # 算术运算
            TestQuestion(
                id="math_arith_001", question="计算: 123 + 456 = ?",
                category="arithmetic", difficulty="easy",
                expected_answers=["579"], max_score=100
            ),
            TestQuestion(
                id="math_arith_002", question="计算: 1000 - 367 = ?",
                category="arithmetic", difficulty="easy",
                expected_answers=["633"], max_score=100
            ),
            TestQuestion(
                id="math_arith_003", question="计算: 25 × 16 = ?",
                category="arithmetic", difficulty="easy",
                expected_answers=["400"], max_score=100
            ),
            TestQuestion(
                id="math_arith_004", question="计算: 144 ÷ 12 = ?",
                category="arithmetic", difficulty="easy",
                expected_answers=["12"], max_score=100
            ),
            TestQuestion(
                id="math_arith_005", question="计算: 15 × 15 = ?",
                category="arithmetic", difficulty="medium",
                expected_answers=["225"], max_score=100
            ),
            TestQuestion(
                id="math_arith_006", question="计算: 99 × 99 = ?",
                category="arithmetic", difficulty="medium",
                expected_answers=["9801"], max_score=100
            ),
            TestQuestion(
                id="math_arith_007", question="计算: 2的10次方等于多少？",
                category="arithmetic", difficulty="medium",
                expected_answers=["1024"], max_score=100
            ),
            TestQuestion(
                id="math_arith_008", question="计算: 3的5次方等于多少？",
                category="arithmetic", difficulty="medium",
                expected_answers=["243"], max_score=100
            ),
            
            # 代数运算
            TestQuestion(
                id="math_alg_001", question="解方程: 2x + 5 = 13，x等于多少？",
                category="algebra", difficulty="easy",
                expected_answers=["4", "x=4", "x = 4"], max_score=100
            ),
            TestQuestion(
                id="math_alg_002", question="解方程: 3x - 7 = 8，x等于多少？",
                category="algebra", difficulty="easy",
                expected_answers=["5", "x=5", "x = 5"], max_score=100
            ),
            TestQuestion(
                id="math_alg_003", question="解方程: x² = 144，x等于多少？",
                category="algebra", difficulty="medium",
                expected_answers=["12", "-12", "±12", "正负12"], max_score=100
            ),
            TestQuestion(
                id="math_alg_004", question="如果 y = 2x + 3，当 x = 5 时，y 等于多少？",
                category="algebra", difficulty="easy",
                expected_answers=["13", "y=13", "y = 13"], max_score=100
            ),
            TestQuestion(
                id="math_alg_005", question="解方程: 5x + 3 = 2x + 12，x等于多少？",
                category="algebra", difficulty="medium",
                expected_answers=["3", "x=3", "x = 3"], max_score=100
            ),
            TestQuestion(
                id="math_alg_006", question="解不等式: 2x - 4 > 6，x的范围是什么？",
                category="algebra", difficulty="hard",
                expected_answers=["x>5", "x > 5", "大于5"], max_score=100
            ),
            
            # 几何运算
            TestQuestion(
                id="math_geo_001", question="一个三角形的底是6厘米，高是4厘米，面积是多少平方厘米？",
                category="geometry", difficulty="easy",
                expected_answers=["12"], max_score=100
            ),
            TestQuestion(
                id="math_geo_002", question="一个圆的半径是5厘米，求面积（π取3.14）",
                category="geometry", difficulty="medium",
                expected_answers=["78.5", "78.5平方厘米"], max_score=100
            ),
            TestQuestion(
                id="math_geo_003", question="一个正方形的边长是8厘米，周长是多少厘米？",
                category="geometry", difficulty="easy",
                expected_answers=["32"], max_score=100
            ),
            TestQuestion(
                id="math_geo_004", question="一个长方形的长是10厘米，宽是6厘米，面积是多少平方厘米？",
                category="geometry", difficulty="easy",
                expected_answers=["60"], max_score=100
            ),
            TestQuestion(
                id="math_geo_005", question="一个圆的周长是31.4厘米（π取3.14），半径是多少厘米？",
                category="geometry", difficulty="medium",
                expected_answers=["5"], max_score=100
            ),
            TestQuestion(
                id="math_geo_006", question="直角三角形的两条直角边分别是3和4，斜边是多少？",
                category="geometry", difficulty="medium",
                expected_answers=["5"], max_score=100
            ),
            
            # 数列与模式
            TestQuestion(
                id="math_seq_001", question="斐波那契数列第10项是多少？（数列从1,1开始）",
                category="sequence", difficulty="hard",
                expected_answers=["55"], max_score=100
            ),
            TestQuestion(
                id="math_seq_002", question="数列 1, 2, 4, 8, 16, ? 的下一项是什么？",
                category="sequence", difficulty="easy",
                expected_answers=["32"], max_score=100
            ),
            TestQuestion(
                id="math_seq_003", question="数列 2, 6, 12, 20, 30, ? 的下一项是什么？",
                category="sequence", difficulty="medium",
                expected_answers=["42"], max_score=100
            ),
        ]
        
        # ==================== 代码能力测试 ====================
        self.datasets["code"] = [
            # Python基础
            TestQuestion(
                id="code_py_001", question="写一个Python函数，计算列表中所有数字的和",
                category="python_basic", difficulty="easy",
                expected_answers=[],
                keywords=["def", "sum", "return", "for"],
                scoring_rubric={"has_function": 25, "has_loop": 25, "correct_logic": 25, "has_return": 25},
                max_score=100
            ),
            TestQuestion(
                id="code_py_002", question="写一个Python函数，判断一个数是否为偶数",
                category="python_basic", difficulty="easy",
                expected_answers=[],
                keywords=["def", "if", "%", "return", "True", "False"],
                scoring_rubric={"has_function": 20, "has_condition": 20, "modulo_op": 30, "returns_bool": 30},
                max_score=100
            ),
            TestQuestion(
                id="code_py_003", question="写一个Python函数，反转一个字符串",
                category="python_basic", difficulty="easy",
                expected_answers=[],
                keywords=["def", "return", "[::-1]", "reversed"],
                scoring_rubric={"has_function": 25, "reverse_method": 50, "correct_return": 25},
                max_score=100
            ),
            TestQuestion(
                id="code_py_004", question="写一个Python类表示一个简单的银行账户，包含存款和取款方法",
                category="python_oop", difficulty="medium",
                expected_answers=[],
                keywords=["class", "def", "self", "__init__", "deposit", "withdraw"],
                scoring_rubric={"has_class": 20, "has_init": 20, "has_deposit": 20, "has_withdraw": 20, "balance_track": 20},
                max_score=100
            ),
            
            # 算法
            TestQuestion(
                id="code_alg_001", question="写一个函数实现二分查找算法",
                category="algorithm", difficulty="medium",
                expected_answers=[],
                keywords=["def", "while", "mid", "return", "left", "right"],
                scoring_rubric={"has_function": 15, "binary_search_logic": 40, "handles_boundaries": 25, "correct_return": 20},
                max_score=100
            ),
            TestQuestion(
                id="code_alg_002", question="写一个函数判断一个数是否为质数",
                category="algorithm", difficulty="medium",
                expected_answers=[],
                keywords=["def", "if", "for", "range", "%", "return"],
                scoring_rubric={"has_function": 15, "handles_edge_cases": 20, "divisibility_check": 35, "correct_logic": 30},
                max_score=100
            ),
            TestQuestion(
                id="code_alg_003", question="写一个函数实现冒泡排序",
                category="algorithm", difficulty="medium",
                expected_answers=[],
                keywords=["def", "for", "range", "if", "swap"],
                scoring_rubric={"has_function": 15, "nested_loops": 25, "swap_logic": 30, "correct_algorithm": 30},
                max_score=100
            ),
            TestQuestion(
                id="code_alg_004", question="写一个函数计算斐波那契数列的第n项（使用递归）",
                category="algorithm", difficulty="medium",
                expected_answers=[],
                keywords=["def", "if", "return", "fibonacci", "n-1", "n-2"],
                scoring_rubric={"has_function": 15, "base_case": 25, "recursive_call": 35, "correct_logic": 25},
                max_score=100
            ),
            
            # 数据结构
            TestQuestion(
                id="code_ds_001", question="写一个简单的栈（Stack）类，包含push、pop方法",
                category="data_structure", difficulty="medium",
                expected_answers=[],
                keywords=["class", "def", "self", "push", "pop", "list", "__init__"],
                scoring_rubric={"has_class": 15, "has_init": 15, "has_push": 25, "has_pop": 25, "correct_logic": 20},
                max_score=100
            ),
            TestQuestion(
                id="code_ds_002", question="写一个函数实现队列（Queue）的基本操作",
                category="data_structure", difficulty="medium",
                expected_answers=[],
                keywords=["class", "def", "enqueue", "dequeue", "self", "list"],
                scoring_rubric={"has_class": 15, "enqueue_method": 30, "dequeue_method": 30, "correct_logic": 25},
                max_score=100
            ),
            
            # 代码理解
            TestQuestion(
                id="code_concept_001", question="HTTP状态码200表示什么？",
                category="concept", difficulty="easy",
                expected_answers=["成功", "OK", "请求成功", "成功响应"],
                max_score=100
            ),
            TestQuestion(
                id="code_concept_002", question="快速排序的平均时间复杂度是多少？",
                category="concept", difficulty="medium",
                expected_answers=["O(n log n)", "O(nlogn)", "n log n"],
                max_score=100
            ),
            TestQuestion(
                id="code_concept_003", question="SQL中SELECT语句用于什么？",
                category="concept", difficulty="easy",
                expected_answers=["查询", "检索", "获取数据", "查询数据"],
                max_score=100
            ),
            TestQuestion(
                id="code_concept_004", question="什么是API？",
                category="concept", difficulty="easy",
                expected_answers=["应用程序接口", "接口", "Application Programming Interface"],
                max_score=100
            ),
            TestQuestion(
                id="code_concept_005", question="Git用于什么？",
                category="concept", difficulty="easy",
                expected_answers=["版本控制", "代码管理", "版本管理"],
                max_score=100
            ),
        ]
        
        # ==================== 知识问答测试 ====================
        self.datasets["knowledge"] = [
            # 地理知识
            TestQuestion(
                id="know_geo_001", question="中国的首都是哪个城市？",
                category="geography", difficulty="easy",
                expected_answers=["北京"], max_score=100
            ),
            TestQuestion(
                id="know_geo_002", question="世界上最长的河流是哪条？",
                category="geography", difficulty="medium",
                expected_answers=["尼罗河", "亚马逊河"],  # 有争议，两个都可接受
                max_score=100
            ),
            TestQuestion(
                id="know_geo_003", question="中国的最长河流是哪条？",
                category="geography", difficulty="easy",
                expected_answers=["长江"], max_score=100
            ),
            TestQuestion(
                id="know_geo_004", question="世界上最高的山峰是哪座？",
                category="geography", difficulty="easy",
                expected_answers=["珠穆朗玛峰", "珠峰"], max_score=100
            ),
            
            # 科学知识
            TestQuestion(
                id="know_sci_001", question="光在真空中的传播速度大约是多少？",
                category="science", difficulty="medium",
                expected_answers=["30万公里每秒", "299792458米每秒", "3×10^8米每秒", "30万公里"],
                max_score=100
            ),
            TestQuestion(
                id="know_sci_002", question="水的化学式是什么？",
                category="science", difficulty="easy",
                expected_answers=["H2O", "H₂O"], max_score=100
            ),
            TestQuestion(
                id="know_sci_003", question="地球绕太阳公转一周需要多长时间？",
                category="science", difficulty="easy",
                expected_answers=["一年", "365天", "365.25天"],
                max_score=100
            ),
            TestQuestion(
                id="know_sci_004", question="太阳系中最大的行星是哪颗？",
                category="science", difficulty="easy",
                expected_answers=["木星"], max_score=100
            ),
            TestQuestion(
                id="know_sci_005", question="DNA的全称是什么？",
                category="science", difficulty="medium",
                expected_answers=["脱氧核糖核酸"], max_score=100
            ),
            TestQuestion(
                id="know_sci_006", question="人体最大的器官是什么？",
                category="science", difficulty="easy",
                expected_answers=["皮肤"], max_score=100
            ),
            
            # 历史知识
            TestQuestion(
                id="know_hist_001", question="《红楼梦》的作者是谁？",
                category="history", difficulty="easy",
                expected_answers=["曹雪芹"], max_score=100
            ),
            TestQuestion(
                id="know_hist_002", question="谁发明了电话？",
                category="history", difficulty="easy",
                expected_answers=["贝尔", "亚历山大·贝尔"], max_score=100
            ),
            TestQuestion(
                id="know_hist_003", question="相对论是谁提出的？",
                category="history", difficulty="easy",
                expected_answers=["爱因斯坦"], max_score=100
            ),
            TestQuestion(
                id="know_hist_004", question="第一次世界大战开始于哪一年？",
                category="history", difficulty="medium",
                expected_answers=["1914"], max_score=100
            ),
            
            # 文学知识
            TestQuestion(
                id="know_lit_001", question="《三国演义》的作者是谁？",
                category="literature", difficulty="easy",
                expected_answers=["罗贯中"], max_score=100
            ),
            TestQuestion(
                id="know_lit_002", question="《西游记》的作者是谁？",
                category="literature", difficulty="easy",
                expected_answers=["吴承恩"], max_score=100
            ),
            TestQuestion(
                id="know_lit_003", question="诗仙是指哪位诗人？",
                category="literature", difficulty="easy",
                expected_answers=["李白"], max_score=100
            ),
        ]
        
        # ==================== 逻辑推理测试 ====================
        self.datasets["reasoning"] = [
            # 演绎推理
            TestQuestion(
                id="reason_ded_001", question="如果所有的A都是B，所有的B都是C，那么所有的A都是C吗？",
                category="deduction", difficulty="easy",
                expected_answers=["是", "对", "正确", "是的"],
                max_score=100
            ),
            TestQuestion(
                id="reason_ded_002", question="如果所有的猫都是动物，所有的动物都需要水，那么所有的猫都需要水吗？",
                category="deduction", difficulty="easy",
                expected_answers=["是", "对", "正确", "需要"],
                max_score=100
            ),
            TestQuestion(
                id="reason_ded_003", question="如果下雨，地面会湿。现在下雨了，地面会湿吗？",
                category="deduction", difficulty="easy",
                expected_answers=["会", "会湿", "湿"],
                max_score=100
            ),
            
            # 传递推理
            TestQuestion(
                id="reason_trans_001", question="A比B高，B比C高，谁最高？",
                category="transitive", difficulty="easy",
                expected_answers=["A"],
                max_score=100
            ),
            TestQuestion(
                id="reason_trans_002", question="小明比小红高，小红比小华高，谁最矮？",
                category="transitive", difficulty="easy",
                expected_answers=["小华"],
                max_score=100
            ),
            TestQuestion(
                id="reason_trans_003", question="A比B大2岁，B比C大3岁，A比C大几岁？",
                category="transitive", difficulty="medium",
                expected_answers=["5", "5岁", "五岁"],
                max_score=100
            ),
            
            # 时间推理
            TestQuestion(
                id="reason_time_001", question="如果今天是星期三，那么后天是星期几？",
                category="temporal", difficulty="easy",
                expected_answers=["星期五", "周五", "五"],
                max_score=100
            ),
            TestQuestion(
                id="reason_time_002", question="如果今天是星期三，那么100天后是星期几？",
                category="temporal", difficulty="medium",
                expected_answers=["星期五", "周五", "五"],
                max_score=100
            ),
            
            # 数量推理
            TestQuestion(
                id="reason_quant_001", question="有5个苹果，你拿走了3个，你现在有几个苹果？",
                category="quantitative", difficulty="easy",
                expected_answers=["3", "三个"],
                max_score=100
            ),
            TestQuestion(
                id="reason_quant_002", question="一个房间有4个角，每个角有1只猫，每只猫前面有3只猫，房间里一共有几只猫？",
                category="quantitative", difficulty="hard",
                expected_answers=["4", "四"],
                max_score=100
            ),
            
            # 模式识别
            TestQuestion(
                id="reason_pattern_001", question="小明在小红的左边，小红在小华的左边，小明在小华的哪边？",
                category="spatial", difficulty="medium",
                expected_answers=["左边", "左"],
                max_score=100
            ),
            
            # 逻辑谜题
            TestQuestion(
                id="reason_puzzle_001", question="甲说乙在说谎，乙说丙在说谎，丙说甲和乙都在说谎。谁在说真话？",
                category="puzzle", difficulty="hard",
                expected_answers=["乙"],
                max_score=100
            ),
        ]
        
        # ==================== 创造性写作测试 ====================
        self.datasets["creativity"] = [
            # 诗歌创作
            TestQuestion(
                id="creat_poem_001", question="请写一首关于春天的短诗（4-6行）",
                category="poetry", difficulty="medium",
                expected_answers=[],
                scoring_rubric={"length_appropriate": 20, "has_rhythm": 25, "spring_theme": 25, "creativity": 30},
                max_score=100,
                time_limit=60.0
            ),
            TestQuestion(
                id="creat_poem_002", question="请写一首关于大海的现代诗",
                category="poetry", difficulty="medium",
                expected_answers=[],
                scoring_rubric={"length_appropriate": 20, "ocean_imagery": 30, "creativity": 30, "emotional_depth": 20},
                max_score=100,
                time_limit=60.0
            ),
            
            # 故事创作
            TestQuestion(
                id="creat_story_001", question="请编一个关于机器人和人类友谊的短故事",
                category="story", difficulty="medium",
                expected_answers=[],
                scoring_rubric={"story_structure": 25, "character_development": 25, "theme": 25, "creativity": 25},
                max_score=100,
                time_limit=90.0
            ),
            TestQuestion(
                id="creat_story_002", question="请写一个关于时间旅行的小故事",
                category="story", difficulty="hard",
                expected_answers=[],
                scoring_rubric={"story_structure": 25, "time_travel_element": 25, "coherence": 25, "creativity": 25},
                max_score=100,
                time_limit=90.0
            ),
            
            # 想象力
            TestQuestion(
                id="creat_imag_001", question="如果可以时间旅行，你会去哪个时代？为什么？",
                category="imagination", difficulty="easy",
                expected_answers=[],
                scoring_rubric={"clear_choice": 25, "reasoning": 35, "creativity": 25, "expression": 15},
                max_score=100,
                time_limit=45.0
            ),
            TestQuestion(
                id="creat_imag_002", question="如果你能拥有一种超能力，你会选择什么？为什么？",
                category="imagination", difficulty="easy",
                expected_answers=[],
                scoring_rubric={"clear_choice": 25, "reasoning": 35, "creativity": 25, "expression": 15},
                max_score=100,
                time_limit=45.0
            ),
            
            # 设计思维
            TestQuestion(
                id="creat_design_001", question="设计一个未来城市的交通系统",
                category="design", difficulty="hard",
                expected_answers=[],
                scoring_rubric={"feasibility": 25, "innovation": 30, "completeness": 25, "expression": 20},
                max_score=100,
                time_limit=90.0
            ),
            TestQuestion(
                id="creat_design_002", question="设计一种帮助老年人日常生活的智能设备",
                category="design", difficulty="medium",
                expected_answers=[],
                scoring_rubric={"user_needs": 30, "innovation": 25, "practicality": 25, "expression": 20},
                max_score=100,
                time_limit=60.0
            ),
            
            # 描写
            TestQuestion(
                id="creat_desc_001", question="请描写一个下雨天的场景",
                category="description", difficulty="easy",
                expected_answers=[],
                scoring_rubric={"sensory_details": 35, "atmosphere": 30, "creativity": 20, "expression": 15},
                max_score=100,
                time_limit=45.0
            ),
        ]
    
    def _load_external_datasets(self):
        """加载外部数据集文件"""
        if not os.path.exists(DATASETS_PATH):
            os.makedirs(DATASETS_PATH, exist_ok=True)
            self._save_default_datasets()
    
    def _save_default_datasets(self):
        """保存默认数据集到文件"""
        for dim, questions in self.datasets.items():
            filepath = os.path.join(DATASETS_PATH, f"{dim}_questions.json")
            data = {
                "dimension": dim,
                "count": len(questions),
                "questions": [asdict(q) for q in questions]
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_dataset(self, dimension: str) -> List[TestQuestion]:
        """获取指定维度的数据集"""
        return self.datasets.get(dimension, [])
    
    def get_all_datasets(self) -> Dict[str, List[TestQuestion]]:
        """获取所有数据集"""
        return self.datasets


# ============================================================================
# 评分器
# ============================================================================

class AnswerScorer:
    """答案评分器"""
    
    def __init__(self):
        self.creative_keywords = {
            "positive": ["想象", "未来", "美丽", "神奇", "奇妙", "独特", "创新", "精彩", "动人", "优雅"],
            "connectors": ["因为", "所以", "但是", "然而", "而且", "并且", "首先", "其次", "最后"]
        }
    
    def score_answer(
        self,
        question: TestQuestion,
        response: str,
        time_taken: float
    ) -> AnswerResult:
        """评分答案"""
        
        # 根据问题类型选择评分方法
        if question.expected_answers:
            # 有标准答案的问题
            return self._score_exact_match(question, response, time_taken)
        elif question.keywords:
            # 代码类问题，检查关键词
            return self._score_code_question(question, response, time_taken)
        else:
            # 创造性问题
            return self._score_creative_question(question, response, time_taken)
    
    def _score_exact_match(
        self,
        question: TestQuestion,
        response: str,
        time_taken: float
    ) -> AnswerResult:
        """精确匹配评分"""
        response_lower = response.lower().strip()
        is_correct = False
        match_type = "none"
        matched_answer = None
        
        for expected in question.expected_answers:
            expected_lower = expected.lower()
            
            # 完全匹配
            if expected_lower == response_lower:
                is_correct = True
                match_type = "exact"
                matched_answer = expected
                break
            
            # 包含匹配
            if expected_lower in response_lower:
                is_correct = True
                match_type = "partial"
                matched_answer = expected
                break
        
        # 计算分数
        if match_type == "exact":
            score = question.max_score
        elif match_type == "partial":
            score = question.max_score * 0.9
        else:
            score = 0.0
        
        return AnswerResult(
            question_id=question.id,
            question=question.question,
            expected=question.expected_answers,
            response=response[:500],  # 限制长度
            is_correct=is_correct,
            score=score,
            max_score=question.max_score,
            time_taken=time_taken,
            match_type=match_type,
            details={"matched_answer": matched_answer}
        )
    
    def _score_code_question(
        self,
        question: TestQuestion,
        response: str,
        time_taken: float
    ) -> AnswerResult:
        """代码问题评分"""
        if not question.scoring_rubric:
            # 使用关键词评分
            keywords_found = [kw for kw in question.keywords if kw.lower() in response.lower()]
            score = (len(keywords_found) / len(question.keywords)) * question.max_score if question.keywords else 50.0
            is_correct = score >= 60.0
        else:
            # 使用评分细则
            score = 0.0
            details = {}
            
            for criterion, points in question.scoring_rubric.items():
                criterion_score = self._evaluate_code_criterion(criterion, response)
                criterion_points = points * criterion_score
                score += criterion_points
                details[criterion] = criterion_points
            
            is_correct = score >= 60.0
        
        return AnswerResult(
            question_id=question.id,
            question=question.question,
            expected=question.expected_answers,
            response=response[:1000],
            is_correct=is_correct,
            score=score,
            max_score=question.max_score,
            time_taken=time_taken,
            match_type="keyword",
            details={"keywords_checked": question.keywords}
        )
    
    def _evaluate_code_criterion(self, criterion: str, response: str) -> float:
        """评估代码准则"""
        response_lower = response.lower()
        
        criterion_checks = {
            "has_function": lambda r: "def " in r,
            "has_loop": lambda r: any(kw in r for kw in ["for ", "while "]),
            "has_condition": lambda r: "if " in r,
            "has_return": lambda r: "return" in r,
            "has_class": lambda r: "class " in r,
            "has_init": lambda r: "__init__" in r,
            "correct_logic": lambda r: len(r) > 50,  # 简单检查
            "correct_return": lambda r: "return" in r,
            "has_deposit": lambda r: "deposit" in r.lower(),
            "has_withdraw": lambda r: "withdraw" in r.lower(),
            "balance_track": lambda r: "balance" in r.lower() or "amount" in r.lower(),
            "binary_search_logic": lambda r: "mid" in r.lower() and ("left" in r.lower() or "right" in r.lower()),
            "handles_boundaries": lambda r: "while" in r.lower() or "if" in r.lower(),
            "handles_edge_cases": lambda r: "if" in r.lower() and ("return" in r.lower() or "==" in r.lower()),
            "divisibility_check": lambda r: "%" in r or "//" in r,
            "nested_loops": lambda r: r.count("for ") >= 2 or r.count("while ") >= 2,
            "swap_logic": lambda r: "swap" in r.lower() or ("temp" in r.lower() and "=" in r),
            "base_case": lambda r: "if" in r.lower() and ("return" in r.lower() or "==" in r.lower()),
            "recursive_call": lambda r: "def" in r.lower() and "(" in r,
            "enqueue_method": lambda r: "enqueue" in r.lower() or "push" in r.lower() or "append" in r.lower(),
            "dequeue_method": lambda r: "dequeue" in r.lower() or "pop" in r.lower(),
        }
        
        check_func = criterion_checks.get(criterion)
        if check_func:
            return 1.0 if check_func(response_lower) else 0.0
        
        return 0.5  # 未知准则，给一半分
    
    def _score_creative_question(
        self,
        question: TestQuestion,
        response: str,
        time_taken: float
    ) -> AnswerResult:
        """创造性问题评分"""
        score = 0.0
        details = {}
        
        if question.scoring_rubric:
            for criterion, points in question.scoring_rubric.items():
                criterion_score = self._evaluate_creative_criterion(criterion, response, question.category)
                criterion_points = points * criterion_score
                score += criterion_points
                details[criterion] = criterion_points
        else:
            # 默认评分方法
            score = self._default_creative_score(response)
        
        is_correct = score >= 50.0
        
        return AnswerResult(
            question_id=question.id,
            question=question.question,
            expected=[],
            response=response[:1000],
            is_correct=is_correct,
            score=score,
            max_score=question.max_score,
            time_taken=time_taken,
            match_type="semantic",
            details=details
        )
    
    def _evaluate_creative_criterion(self, criterion: str, response: str, category: str) -> float:
        """评估创造性准则"""
        criterion_checks = {
            "length_appropriate": lambda r: 0.5 if len(r) < 50 else (1.0 if len(r) < 500 else 0.7),
            "has_rhythm": lambda r: 1.0 if len(r.split('\n')) >= 3 else 0.5,
            "spring_theme": lambda r: 1.0 if any(w in r for w in ["春", "花", "绿", "温暖", "生机"]) else 0.3,
            "ocean_imagery": lambda r: 1.0 if any(w in r for w in ["海", "浪", "蓝", "深", "广阔"]) else 0.3,
            "creativity": lambda r: 1.0 if any(w in r for w in self.creative_keywords["positive"]) else 0.5,
            "emotional_depth": lambda r: 1.0 if any(w in r for w in ["感", "情", "心", "思念", "回忆"]) else 0.5,
            "story_structure": lambda r: 1.0 if any(w in r for w in ["开始", "然后", "最后", "起初", "后来"]) else 0.5,
            "character_development": lambda r: 1.0 if len([w for w in ["他", "她", "它", "我", "你"] if w in r]) >= 3 else 0.5,
            "theme": lambda r: 1.0 if len(r) > 100 else 0.5,
            "time_travel_element": lambda r: 1.0 if any(w in r for w in ["时间", "过去", "未来", "穿越", "时代"]) else 0.3,
            "coherence": lambda r: 1.0 if any(w in r for w in self.creative_keywords["connectors"]) else 0.5,
            "clear_choice": lambda r: 1.0 if len(r) > 20 else 0.5,
            "reasoning": lambda r: 1.0 if any(w in r for w in ["因为", "所以", "由于", "原因", "因此"]) else 0.5,
            "expression": lambda r: 1.0 if len(r) > 50 else 0.5,
            "feasibility": lambda r: 0.5,  # 需要人工评估
            "innovation": lambda r: 1.0 if any(w in r for w in ["创新", "新颖", "独特", "智能", "自动"]) else 0.5,
            "completeness": lambda r: 1.0 if len(r.split('。')) >= 3 else 0.5,
            "user_needs": lambda r: 1.0 if any(w in r for w in ["方便", "简单", "易用", "舒适", "安全"]) else 0.5,
            "practicality": lambda r: 1.0 if len(r) > 100 else 0.5,
            "sensory_details": lambda r: 1.0 if any(w in r for w in ["看", "听", "闻", "触", "感", "声", "味"]) else 0.5,
            "atmosphere": lambda r: 1.0 if any(w in r for w in ["安静", "宁静", "朦胧", "清新", "凉爽"]) else 0.5,
        }
        
        check_func = criterion_checks.get(criterion)
        if check_func:
            return check_func(response)
        
        return 0.5
    
    def _default_creative_score(self, response: str) -> float:
        """默认创造性评分"""
        score = 0.0
        
        # 长度评分（最多30分）
        if len(response) > 200:
            score += 30
        elif len(response) > 100:
            score += 20
        elif len(response) > 50:
            score += 10
        
        # 词汇丰富度（最多30分）
        unique_words = len(set(response.split()))
        if unique_words > 50:
            score += 30
        elif unique_words > 30:
            score += 20
        elif unique_words > 15:
            score += 10
        
        # 创意词汇（最多20分）
        creative_count = sum(1 for w in self.creative_keywords["positive"] if w in response)
        score += min(creative_count * 4, 20)
        
        # 结构性（最多20分）
        if "。" in response or "！" in response or "？" in response:
            score += 10
        if any(w in response for w in self.creative_keywords["connectors"]):
            score += 10
        
        return min(score, 100.0)


# ============================================================================
# 性能基准测试
# ============================================================================

class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, engine=None):
        self.engine = engine
        self.metrics: List[PerformanceMetrics] = []
    
    def set_engine(self, engine):
        """设置引擎"""
        self.engine = engine
    
    def measure_inference_speed(self, prompt: str, max_tokens: int = 100) -> PerformanceMetrics:
        """测量推理速度"""
        if not self.engine:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)
        
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        first_token_time = None
        token_count = 0
        token_times = []
        stdp_updates = 0
        memory_ops = 0
        
        try:
            for chunk in self.engine.stream_process(prompt, max_tokens=max_tokens):
                if hasattr(chunk, 'type'):
                    if chunk.type == "text":
                        if first_token_time is None:
                            first_token_time = time.time()
                        token_count += 1
                        token_times.append(time.time())
                    elif chunk.type == "weight_update":
                        stdp_updates += 1
                    elif chunk.type in ["memory_call", "memory_store"]:
                        memory_ops += 1
        except Exception as e:
            print(f"推理速度测试错误: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        end_memory = process.memory_info().rss / 1024 / 1024
        
        tps = token_count / total_time if total_time > 0 else 0
        ttft = (first_token_time - start_time) if first_token_time else 0
        latency = (token_times[-1] - token_times[0]) / len(token_times) * 1000 if len(token_times) > 1 else 0
        memory_usage = end_memory - start_memory
        cpu_usage = process.cpu_percent()
        
        metrics = PerformanceMetrics(
            tokens_per_second=tps,
            time_to_first_token=ttft,
            total_inference_time=total_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            latency_ms=latency,
            throughput=token_count,
            stdp_updates=stdp_updates,
            memory_operations=memory_ops
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def measure_memory_performance(self, queries: List[str]) -> Dict:
        """测量记忆性能"""
        if not self.engine or not hasattr(self.engine, 'memory') or not self.engine.memory:
            return {"error": "Memory system not available"}
        
        results = []
        for query in queries:
            start_time = time.time()
            try:
                memories = self.engine.memory.search(query, top_k=5)
                search_time = time.time() - start_time
                results.append({
                    "query": query,
                    "search_time_ms": search_time * 1000,
                    "results_found": len(memories)
                })
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e)
                })
        
        avg_time = sum(r.get("search_time_ms", 0) for r in results) / len(results) if results else 0
        
        return {
            "queries_tested": len(queries),
            "avg_search_time_ms": avg_time,
            "details": results
        }
    
    def measure_latency_distribution(self, prompt: str, iterations: int = 5) -> Dict:
        """测量延迟分布"""
        latencies = []
        ttfts = []
        
        for i in range(iterations):
            metrics = self.measure_inference_speed(prompt, max_tokens=50)
            latencies.append(metrics.latency_ms)
            ttfts.append(metrics.time_to_first_token)
            time.sleep(0.5)  # 冷却
        
        return {
            "iterations": iterations,
            "latency": {
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
                "avg": statistics.mean(latencies) if latencies else 0,
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "time_to_first_token": {
                "min": min(ttfts) if ttfts else 0,
                "max": max(ttfts) if ttfts else 0,
                "avg": statistics.mean(ttfts) if ttfts else 0,
                "std": statistics.stdev(ttfts) if len(ttfts) > 1 else 0
            }
        }
    
    def run_full_performance_test(self) -> Dict:
        """运行完整性能测试"""
        print("\n" + "=" * 60)
        print("性能基准测试")
        print("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "inference_speed": None,
            "memory_performance": None,
            "latency_distribution": None,
            "streaming_performance": None
        }
        
        # 推理速度测试
        print("\n[1/4] 推理速度测试...")
        results["inference_speed"] = asdict(
            self.measure_inference_speed("请解释什么是人工智能？", max_tokens=100)
        )
        print(f"  ✓ Tokens/s: {results['inference_speed']['tokens_per_second']:.1f}")
        print(f"  ✓ 首Token延迟: {results['inference_speed']['time_to_first_token']:.3f}s")
        
        # 记忆性能测试
        print("\n[2/4] 记忆性能测试...")
        results["memory_performance"] = self.measure_memory_performance([
            "人工智能", "机器学习", "神经网络", "深度学习", "自然语言处理"
        ])
        if "avg_search_time_ms" in results["memory_performance"]:
            print(f"  ✓ 平均搜索时间: {results['memory_performance']['avg_search_time_ms']:.2f}ms")
        
        # 延迟分布测试
        print("\n[3/4] 延迟分布测试...")
        results["latency_distribution"] = self.measure_latency_distribution(
            "什么是机器学习？", iterations=3
        )
        print(f"  ✓ 平均延迟: {results['latency_distribution']['latency']['avg']:.2f}ms")
        
        # 流式处理性能
        print("\n[4/4] 流式处理性能...")
        results["streaming_performance"] = self._measure_streaming_performance()
        
        print("\n" + "=" * 60)
        
        return results
    
    def _measure_streaming_performance(self) -> Dict:
        """测量流式处理性能"""
        if not self.engine:
            return {"error": "Engine not available"}
        
        chunk_intervals = []
        chunk_count = 0
        
        try:
            last_chunk_time = None
            for chunk in self.engine.stream_process("请详细解释量子计算的基本原理", max_tokens=150):
                if hasattr(chunk, 'type') and chunk.type == "text":
                    now = time.time()
                    if last_chunk_time:
                        chunk_intervals.append((now - last_chunk_time) * 1000)
                    last_chunk_time = now
                    chunk_count += 1
        except Exception as e:
            return {"error": str(e)}
        
        return {
            "total_chunks": chunk_count,
            "avg_chunk_interval_ms": statistics.mean(chunk_intervals) if chunk_intervals else 0,
            "chunk_interval_std": statistics.stdev(chunk_intervals) if len(chunk_intervals) > 1 else 0,
            "min_interval_ms": min(chunk_intervals) if chunk_intervals else 0,
            "max_interval_ms": max(chunk_intervals) if chunk_intervals else 0
        }


# ============================================================================
# 基准测试套件
# ============================================================================

class BenchmarkSuite:
    """完整基准测试套件"""
    
    # GLM-5基准线（用于对比）
    BASELINE = {
        "math": {"accuracy": 0.68, "f1": 0.65},
        "code": {"accuracy": 0.72, "f1": 0.70},
        "knowledge": {"accuracy": 0.85, "f1": 0.83},
        "reasoning": {"accuracy": 0.78, "f1": 0.75},
        "creativity": {"accuracy": 0.70, "f1": 0.68}
    }
    
    def __init__(self, engine=None, output_path: str = None):
        self.engine = engine
        self.output_path = output_path or OUTPUT_PATH
        self.datasets = TestDatasets()
        self.scorer = AnswerScorer()
        self.performance = PerformanceBenchmark(engine)
        
        # 结果存储
        self.dimension_results: Dict[str, DimensionResult] = {}
        self.history: List[Dict] = []
    
    def set_engine(self, engine):
        """设置引擎"""
        self.engine = engine
        self.performance.set_engine(engine)
    
    def _generate_response(self, question: str, max_tokens: int = 150) -> Tuple[str, float]:
        """生成回答"""
        if not self.engine:
            return "", 0.0
        
        start_time = time.time()
        response = ""
        
        try:
            for chunk in self.engine.stream_process(question, max_tokens=max_tokens, search_wiki=True):
                if hasattr(chunk, 'type') and chunk.type == "text":
                    response += chunk.content
        except Exception as e:
            print(f"生成回答错误: {e}")
        
        time_taken = time.time() - start_time
        return response, time_taken
    
    def assess_dimension(self, dimension: str) -> DimensionResult:
        """评估单个维度"""
        questions = self.datasets.get_dataset(dimension)
        if not questions:
            return DimensionResult(
                dimension=dimension,
                dimension_name=dimension,
                total_score=0,
                max_score=0,
                accuracy=0,
                f1_score=0,
                precision=0,
                recall=0,
                avg_time=0
            )
        
        print(f"\n评估维度: {dimension} ({len(questions)}题)")
        print("-" * 40)
        
        results: List[AnswerResult] = []
        correct_count = 0
        total_score = 0
        total_time = 0
        
        # 按子类别统计
        subcategories: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0, "score": 0})
        
        for i, q in enumerate(questions, 1):
            # 根据问题类型调整max_tokens
            max_tokens = 300 if q.category in ["poetry", "story", "design", "description"] else 150
            
            response, time_taken = self._generate_response(q.question, max_tokens=max_tokens)
            result = self.scorer.score_answer(q, response, time_taken)
            results.append(result)
            
            total_score += result.score
            total_time += time_taken
            
            if result.is_correct:
                correct_count += 1
            
            # 更新子类别统计
            subcategories[q.category]["total"] += 1
            subcategories[q.category]["score"] += result.score
            if result.is_correct:
                subcategories[q.category]["correct"] += 1
            
            # 打印进度
            status = "✓" if result.is_correct else "✗"
            print(f"  [{i}/{len(questions)}] {status} {q.id}: {result.score:.0f}分 ({time_taken:.1f}s)")
        
        # 计算指标
        total_questions = len(questions)
        max_possible_score = total_questions * 100
        accuracy = correct_count / total_questions if total_questions > 0 else 0
        
        # 计算Precision, Recall, F1
        # 这里简化处理：correct = TP, incorrect = FP, missed = FN
        tp = correct_count
        fp = total_questions - correct_count
        fn = 0  # 所有问题都尝试回答了
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 构建结果
        dimension_result = DimensionResult(
            dimension=dimension,
            dimension_name=self._get_dimension_name(dimension),
            total_score=total_score,
            max_score=max_possible_score,
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            avg_time=total_time / total_questions if total_questions > 0 else 0,
            questions=results,
            subcategories=dict(subcategories)
        )
        
        self.dimension_results[dimension] = dimension_result
        
        print(f"\n  结果: {correct_count}/{total_questions} 正确, 准确率: {accuracy*100:.1f}%, F1: {f1:.3f}")
        
        return dimension_result
    
    def _get_dimension_name(self, dimension: str) -> str:
        """获取维度名称"""
        names = {
            "math": "数学能力",
            "code": "代码能力",
            "knowledge": "知识问答",
            "reasoning": "逻辑推理",
            "creativity": "创造性写作"
        }
        return names.get(dimension, dimension)
    
    def run_full_assessment(self, dimensions: List[str] = None) -> BenchmarkReport:
        """运行完整评估"""
        dimensions = dimensions or ["math", "code", "knowledge", "reasoning", "creativity"]
        
        print("\n" + "=" * 60)
        print("DigBrain 多维度能力基准测试")
        print("=" * 60)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试维度: {', '.join(dimensions)}")
        print("=" * 60)
        
        start_time = time.time()
        
        # 评估各维度
        for dim in dimensions:
            self.assess_dimension(dim)
        
        # 运行性能测试
        print("\n" + "=" * 60)
        print("性能基准测试")
        print("=" * 60)
        performance_results = self.performance.run_full_performance_test()
        
        total_time = time.time() - start_time
        
        # 计算总分
        total_score = sum(r.total_score for r in self.dimension_results.values())
        max_score = sum(r.max_score for r in self.dimension_results.values())
        overall_score = total_score / max_score * 100 if max_score > 0 else 0
        
        # 与基准对比
        comparison = {}
        for dim, result in self.dimension_results.items():
            baseline = self.BASELINE.get(dim, {"accuracy": 0.7, "f1": 0.68})
            comparison[dim] = {
                "accuracy_diff": result.accuracy - baseline["accuracy"],
                "f1_diff": result.f1_score - baseline["f1"],
                "vs_baseline": "above" if result.accuracy > baseline["accuracy"] else "below"
            }
        
        # 构建报告
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            model_name="DigBrain-Engine",
            total_time=total_time,
            overall_score=overall_score,
            dimensions=self.dimension_results,
            performance=performance_results.get("inference_speed", {}),
            comparison=comparison,
            history_comparison=self._load_history()[-5:]  # 最近5次
        )
        
        # 保存结果
        self._save_report(report)
        
        # 打印摘要
        self._print_summary(report)
        
        return report
    
    def _load_history(self) -> List[Dict]:
        """加载历史记录"""
        history_file = os.path.join(self.output_path, "benchmark_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save_report(self, report: BenchmarkReport):
        """保存报告"""
        os.makedirs(self.output_path, exist_ok=True)
        
        # 保存当前报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.output_path, f"benchmark_{timestamp}.json")
        
        # 转换为可序列化格式
        report_dict = {
            "timestamp": report.timestamp,
            "model_name": report.model_name,
            "total_time": report.total_time,
            "overall_score": report.overall_score,
            "dimensions": {
                dim: {
                    "dimension": r.dimension,
                    "dimension_name": r.dimension_name,
                    "total_score": r.total_score,
                    "max_score": r.max_score,
                    "accuracy": r.accuracy,
                    "f1_score": r.f1_score,
                    "precision": r.precision,
                    "recall": r.recall,
                    "avg_time": r.avg_time,
                    "subcategories": r.subcategories
                }
                for dim, r in report.dimensions.items()
            },
            "performance": report.performance,
            "comparison": report.comparison,
            "history_comparison": report.history_comparison
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n报告已保存: {report_file}")
        
        # 更新历史记录
        history_file = os.path.join(self.output_path, "benchmark_history.json")
        history = self._load_history()
        history.append({
            "timestamp": report.timestamp,
            "overall_score": report.overall_score,
            "dimensions": {dim: {"accuracy": r.accuracy, "f1": r.f1_score} 
                          for dim, r in report.dimensions.items()}
        })
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history[-20:], f, ensure_ascii=False, indent=2)  # 保留最近20次
    
    def _print_summary(self, report: BenchmarkReport):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("评估结果摘要")
        print("=" * 60)
        
        print(f"\n总耗时: {report.total_time:.1f}秒")
        print(f"综合得分: {report.overall_score:.1f}/100")
        
        print("\n各维度得分:")
        print("-" * 50)
        
        for dim, result in report.dimensions.items():
            acc = result.accuracy
            f1 = result.f1_score
            comp = report.comparison.get(dim, {})
            diff = comp.get("accuracy_diff", 0)
            
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"  {result.dimension_name:8s} [{bar}] {acc*100:5.1f}% (F1:{f1:.2f}) [{diff*100:+.1f}%]")
        
        print("-" * 50)
        
        # 对比基准
        avg_diff = sum(c.get("accuracy_diff", 0) for c in report.comparison.values()) / len(report.comparison)
        if avg_diff > 0:
            print(f"\n🎉 整体超越基准 {avg_diff*100:.1f}%!")
        else:
            print(f"\n📊 距离基准差 {-avg_diff*100:.1f}%")
        
        print("=" * 60)


# ============================================================================
# 报告生成器
# ============================================================================

class ReportGenerator:
    """可视化报告生成器"""
    
    def __init__(self, output_path: str = None):
        self.output_path = output_path or OUTPUT_PATH
    
    def generate_html_report(self, report: BenchmarkReport, save_path: str = None) -> str:
        """生成HTML可视化报告"""
        save_path = save_path or os.path.join(self.output_path, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DigBrain 基准测试报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid #0f3460;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            color: #e94560;
            margin-bottom: 10px;
        }}
        .header .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
        .overall-score {{
            background: linear-gradient(135deg, #0f3460, #1a1a2e);
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .overall-score .score {{
            font-size: 4em;
            color: #e94560;
            font-weight: bold;
        }}
        .overall-score .label {{
            color: #888;
            font-size: 1.2em;
        }}
        .dimensions {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .dimension-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }}
        .dimension-card:hover {{
            transform: translateY(-5px);
        }}
        .dimension-card h3 {{
            color: #e94560;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .score-bar {{
            background: #333;
            border-radius: 10px;
            height: 20px;
            margin: 15px 0;
            overflow: hidden;
        }}
        .score-bar-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }}
        .metrics {{
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
            font-size: 0.9em;
        }}
        .metric {{
            text-align: center;
        }}
        .metric .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #4ecca3;
        }}
        .metric .label {{
            color: #888;
            font-size: 0.8em;
        }}
        .performance {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        .performance h2 {{
            color: #e94560;
            margin-bottom: 20px;
        }}
        .performance-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .perf-item {{
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        .perf-item .value {{
            font-size: 1.8em;
            color: #4ecca3;
            font-weight: bold;
        }}
        .perf-item .label {{
            color: #888;
            font-size: 0.85em;
            margin-top: 5px;
        }}
        .comparison {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
        }}
        .comparison h2 {{
            color: #e94560;
            margin-bottom: 20px;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        .comparison-table th {{
            color: #888;
            font-weight: normal;
        }}
        .positive {{ color: #4ecca3; }}
        .negative {{ color: #e94560; }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 DigBrain 基准测试报告</h1>
            <div class="timestamp">测试时间: {report.timestamp}</div>
        </div>
        
        <div class="overall-score">
            <div class="label">综合得分</div>
            <div class="score">{report.overall_score:.1f}</div>
            <div class="label">/ 100</div>
        </div>
        
        <div class="dimensions">
            {self._generate_dimension_cards(report)}
        </div>
        
        <div class="performance">
            <h2>⚡ 性能指标</h2>
            <div class="performance-grid">
                {self._generate_performance_items(report)}
            </div>
        </div>
        
        <div class="comparison">
            <h2>📊 与基准对比</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>维度</th>
                        <th>准确率</th>
                        <th>F1分数</th>
                        <th>对比基准</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_comparison_rows(report)}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>DigBrain Benchmark System v1.0</p>
            <p>总耗时: {report.total_time:.1f}秒</p>
        </div>
    </div>
</body>
</html>"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已生成: {save_path}")
        return save_path
    
    def _generate_dimension_cards(self, report: BenchmarkReport) -> str:
        """生成维度卡片"""
        cards = []
        for dim, result in report.dimensions.items():
            bar_color = "#4ecca3" if result.accuracy >= 0.7 else "#e94560"
            cards.append(f"""
            <div class="dimension-card">
                <h3>{result.dimension_name}</h3>
                <div class="score-bar">
                    <div class="score-bar-fill" style="width: {result.accuracy*100}%; background: {bar_color};"></div>
                </div>
                <div class="metrics">
                    <div class="metric">
                        <div class="value">{result.accuracy*100:.1f}%</div>
                        <div class="label">准确率</div>
                    </div>
                    <div class="metric">
                        <div class="value">{result.f1_score:.2f}</div>
                        <div class="label">F1分数</div>
                    </div>
                    <div class="metric">
                        <div class="value">{result.avg_time:.1f}s</div>
                        <div class="label">平均用时</div>
                    </div>
                </div>
            </div>
            """)
        return "\n".join(cards)
    
    def _generate_performance_items(self, report: BenchmarkReport) -> str:
        """生成性能项"""
        perf = report.performance
        items = []
        
        if isinstance(perf, dict):
            items.append(f"""
            <div class="perf-item">
                <div class="value">{perf.get('tokens_per_second', 0):.1f}</div>
                <div class="label">Tokens/秒</div>
            </div>
            """)
            items.append(f"""
            <div class="perf-item">
                <div class="value">{perf.get('time_to_first_token', 0):.2f}s</div>
                <div class="label">首Token延迟</div>
            </div>
            """)
            items.append(f"""
            <div class="perf-item">
                <div class="value">{perf.get('total_inference_time', 0):.1f}s</div>
                <div class="label">总推理时间</div>
            </div>
            """)
            items.append(f"""
            <div class="perf-item">
                <div class="value">{perf.get('memory_usage_mb', 0):.1f}MB</div>
                <div class="label">内存使用</div>
            </div>
            """)
        
        return "\n".join(items)
    
    def _generate_comparison_rows(self, report: BenchmarkReport) -> str:
        """生成对比行"""
        rows = []
        for dim, result in report.dimensions.items():
            comp = report.comparison.get(dim, {})
            diff = comp.get("accuracy_diff", 0)
            diff_class = "positive" if diff >= 0 else "negative"
            diff_str = f"+{diff*100:.1f}%" if diff >= 0 else f"{diff*100:.1f}%"
            
            rows.append(f"""
            <tr>
                <td>{result.dimension_name}</td>
                <td>{result.accuracy*100:.1f}%</td>
                <td>{result.f1_score:.3f}</td>
                <td class="{diff_class}">{diff_str}</td>
            </tr>
            """)
        return "\n".join(rows)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 运行基准测试"""
    print("\n" + "=" * 60)
    print("DigBrain 基准测试系统")
    print("=" * 60)
    
    # 尝试导入引擎
    try:
        from core.brain_engine import BrainLikeStreamingEngine
        
        # 初始化引擎
        engine = BrainLikeStreamingEngine(
            refresh_rate=60,
            enable_stdp=True,
            enable_memory=True,
            enable_wiki=True
        )
        
        print("\n加载模型...")
        if not engine.load_models():
            print("❌ 模型加载失败")
            return
        
        # 创建测试套件
        benchmark = BenchmarkSuite(engine)
        
        # 运行完整评估
        report = benchmark.run_full_assessment()
        
        # 生成HTML报告
        generator = ReportGenerator()
        generator.generate_html_report(report)
        
        print("\n✅ 基准测试完成！")
        
    except ImportError as e:
        print(f"⚠️ 无法导入引擎: {e}")
        print("使用模拟模式运行测试...")
        
        # 使用模拟模式
        benchmark = BenchmarkSuite()
        # 这里可以添加模拟测试逻辑
        
        print("\n提示: 请确保正确安装了所有依赖")


if __name__ == "__main__":
    main()
