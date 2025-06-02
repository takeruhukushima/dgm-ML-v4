from typing import Dict, List, Optional
import ast
import pandas as pd

class PipelineValidator:
    """パイプラインコードの検証クラス"""
    
    REQUIRED_FUNCTIONS = [
        'preprocess_data',
        'create_model_for_optuna',
        'train_final_model',
        'evaluate_model'
    ]
    
    REQUIRED_GLOBALS = [
        'MODEL_TYPE',
        'FE_STEPS_DESCRIPTION'
    ]
    
    def __init__(self):
        self.validation_errors: List[str] = []
    
    def validate_pipeline_code(self, code: str) -> bool:
        """パイプラインコードが必要な要件を満たしているか検証"""
        self.validation_errors = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self.validation_errors.append(f"Syntax error in code: {e}")
            return False
            
        # 必要な関数の存在チェック
        found_functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                found_functions.add(node.name)
                
        for required_func in self.REQUIRED_FUNCTIONS:
            if required_func not in found_functions:
                self.validation_errors.append(f"Missing required function: {required_func}")
                
        # グローバル変数の存在チェック
        found_globals = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found_globals.add(target.id)
                        
        for required_global in self.REQUIRED_GLOBALS:
            if required_global not in found_globals:
                self.validation_errors.append(f"Missing required global variable: {required_global}")
                
        return len(self.validation_errors) == 0