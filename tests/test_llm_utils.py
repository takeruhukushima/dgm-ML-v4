import pytest
from unittest.mock import patch, MagicMock
from dgm_core.llm_utils import generate_improvement_suggestion, setup_llm
from dgm_core.config import Config

@pytest.fixture
def mock_genai():
    with patch('dgm_core.llm_utils.genai') as mock:
        mock.GenerativeModel.return_value = MagicMock()
        mock.GenerativeModel.return_value.generate_content.return_value = MagicMock(
            text='{"improvement_description": "Test", "improved_code": "code", "expected_improvement": "test"}'
        )
        yield mock

def test_llm_setup(mock_genai):
    """Test LLM initialization"""
    model = setup_llm()
    assert model is not None
    mock_genai.configure.assert_called_once()

def test_improvement_suggestion(mock_genai):
    """Test improvement suggestion generation"""
    task_description = "Binary classification task for Titanic survival prediction"
    current_code = """
def preprocess_data(df):
    return df
    """
    metrics = {"accuracy": 0.75}
    task_config = {"target_column": "Survived"}
    global_config = {}
    
    suggestion = generate_improvement_suggestion(
        task_description,
        current_code,
        metrics,
        task_config,
        global_config
    )
    
    assert suggestion is not None
    assert isinstance(suggestion, str)
    assert len(suggestion) > 0