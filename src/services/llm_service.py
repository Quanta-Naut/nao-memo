import google.generativeai as genai
from src.core.config import Config

class LLMService:
    def __init__(self):
        Config.validate()
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GENERATIVE_MODEL)

    def decide_memory_importance(self, text: str) -> tuple[bool, str]:
        """
        Decides if a piece of information is worth storing in long-term memory.
        Returns: (should_store: bool, reason: str)
        """
        prompt = f"""
        You are a memory manager AI. Your task is to decide if the following user input contains information 
        that is worth storing in long-term memory.
        
        Worth storing: Facts about the user, preferences, specific events, future plans, relationships, 
        setup details, crucial context.
        
        Not worth storing: Casual greetings ("Hi", "How are you"), fleeting thoughts, 
        questions requesting ephemeral information ("What is 2+2?"), or nonsensical input.

        Input: "{text}"

        Return your answer in the following format ONLY:
        DECISION: [YES/NO]
        REASON: [Brief explanation]
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            should_store = "DECISION: YES" in result
            reason = result.split("REASON:")[1].strip() if "REASON:" in result else "No reason provided."
            
            return should_store, reason
        except Exception as e:
            return False, f"LLM Error: {str(e)}"

    def generate_chat_response(self, query: str, context: list[str]) -> str:
        """
        Generates a response to the user query using the retrieved memory context.
        """
        context_str = "\n".join([f"- {m}" for m in context]) if context else "No relevant memories found."
        
        prompt = f"""
        You are a helpful AI assistant with access to the user's long-term memory.
        
        User Query: "{query}"
        
        Relavant Memories (These are facts about the USER. If they say "I", it refers to the USER, not you.):
        {context_str}
        
        Instructions:
        1. Answer the user's query naturally.
        2. The memories provided are about the USER. Do NOT confuse them with your own preferences.
        3. ONLY use the memories if they are directly relevant to the current topic. Do not randomly recite facts.
           (e.g., if user says "Hi", do not say "Hi, you like sushi". Just say "Hi").
        4. If a memory seems to contradict another, prioritize the most recent one or the one that claims to be "only" or "current" preference.
           (Note: The retrieval system might show old memories if they weren't deleted. Use your judgment.)
        5. Do not explicitly mention "I found this in your memory" unless relevant.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"I'm having trouble thinking right now. Error: {str(e)}"

    def check_memory_update(self, new_text: str, existing_memories: list) -> tuple[str, list[int]]:
        """
        Checks if new information conflicts with or updates existing memories.
        Returns: (Action, list_of_ids_to_delete)
        Action: "ADD", "UPDATE", "IGNORE"
        """
        existing_str = "\n".join([f"ID {m.id}: {m.text}" for m in existing_memories])
        
        prompt = f"""
        You are a memory manager.
        
        New Information: "{new_text}"
        
        Existing Memories:
        {existing_str}
        
        Task: Determine how to handle the New Information in relation to Existing Memories.
        - UPDATE: If new info contradicts or supersedes an existing memory.
          * Example: "I like Java" overrides "I like Python".
          * Example: "I ONLY like french fries" overrides ALL other food preferences (sushi, noodles, etc.).
          * If multiple memories conflict, list ALL their IDs in TARGET_IDS.
        - IGNORE: If new info is exactly the same as an existing memory.
        - ADD: If new info is new and doesn't conflict.
        
        Return ONLY:
        ACTION: [ADD/UPDATE/IGNORE]
        TARGET_IDS: [id1, id2] (ids of memories to delete/replace, empty if ADD)
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            action = "ADD"
            if "ACTION: UPDATE" in result:
                action = "UPDATE"
            elif "ACTION: IGNORE" in result:
                action = "IGNORE"
                
            ids = []
            if "TARGET_IDS:" in result:
                ids_str = result.split("TARGET_IDS:")[1].strip()
                import re
                ids = [int(i) for i in re.findall(r'\d+', ids_str)]
                
            return action, ids
        except Exception as e:
            return "ADD", []
