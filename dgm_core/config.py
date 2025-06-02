import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class Config:
    """Global configuration settings"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # LLM Settings
    LLM_MODEL = "gemini-1.5-pro"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 4096
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")