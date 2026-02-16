import sys
import os
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.getcwd())

from src.services.triplet_service import TripletService
from src.services.llm_service import LLMService

def test_triplet_service_fix():
    # Mock LLMService
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.generate_text.return_value = "query1\nquery2"
    
    # Mock EmbeddingService (not used in the specific failing line, but needed for init)
    mock_embedding = MagicMock()
    
    # Initialize TripletService
    service = TripletService(mock_llm, mock_embedding)
    
    # Mock memory object
    mock_memory = MagicMock()
    mock_memory.text = "This is a memory."
    mock_memory.embedding = [0.1, 0.2, 0.3]
    
    # Run the method that was failing
    try:
        # We need at least 3 memories for the method to proceed effectively or catch the early return
        # But we want to test the _generate_queries_for_memory call which happens inside the loop
        
        # Testing private method directly to verify the fix
        queries = service._generate_queries_for_memory("test memory", 2)
        
        print("Successfully generated queries:", queries)
        
        # Verify generate_text was called
        mock_llm.generate_text.assert_called_once()
        print("Verified: LLMService.generate_text was called.")
        
    except AttributeError as e:
        print(f"FAILED: AttributeError caught: {e}")
    except Exception as e:
        print(f"FAILED: Exception caught: {e}")

if __name__ == "__main__":
    test_triplet_service_fix()
