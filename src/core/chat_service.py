from src.core.memory_manager import MemoryManager
from src.services.llm_service import LLMService

class ChatService:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.llm_service = LLMService()

    def chat(self, user_input: str) -> str:
        """
        Orchestrates the chat flow:
        1. Search for relevant memories.
        2. Generate response using LLM + Context.
        """
        # 1. Search for context
        # We retrieve top 3 memories to keep context focused
        results = self.memory_manager.search_memory(user_input, limit=3)
        
        # Extract just the text from the top results, filtering by a relevance threshold if desired
        context = [result[0].text for result in results if result[1] > 0.6]

        # 2. Parallel: Check if this new input should be stored as memory
        # In a real app, this should be async to not block the chat response
        store_result = self.memory_manager.process_input(user_input)
        
        # 3. Generate response
        response = self.llm_service.generate_chat_response(user_input, context)
        
        return response, store_result
