import sys
import os
import traceback

# Add src to path
sys.path.append(os.getcwd())

from src.services.llm_service import LLMService
from src.core.config import Config

def test_llm_service_real():
    print(f"Config.LLM_PROVIDER: '{Config.LLM_PROVIDER}'")
    print(f"Config.GEMINI_API_KEY: '{Config.GEMINI_API_KEY}'")
    
    try:
        service = LLMService()
        print("LLMService instantiated.")
        
        if hasattr(service, 'model'):
            print("LLMService has 'model' attribute.")
            print(f"Model type: {type(service.model)}")
        else:
            print("LLMService DOES NOT have 'model' attribute.")
            
        print("Attempting to generate text...")
        try:
            text = service.generate_text("Test prompt")
            print(f"Generated text: {text}")
        except Exception as e:
            print("Error during generation:")
            traceback.print_exc()
            
    except Exception as e:
        print("Error during instantiation:")
        traceback.print_exc()

if __name__ == "__main__":
    test_llm_service_real()
