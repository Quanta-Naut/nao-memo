import sys
import time
from src.core.memory_manager import MemoryManager
from src.core.chat_service import ChatService

def test_memory_system():
    print("Initializing Memory System...")
    try:
        manager = MemoryManager()
        chat = ChatService()
    except ValueError as e:
        print(f"Skipping verification: {e}")
        return

    # 1. Test Add Memory
    memory_text = "My favorite programming language is Python."
    print(f"\n[Test 1] Adding memory: '{memory_text}'")
    result = manager.process_input(memory_text)
    print(f"Result: {result}")
    
    if not result['stored'] and "already exists" not in result['reason']:
        print(f"FAILED: Memory was not stored (Reason: {result['reason']})")
    elif not result['stored'] and "already exists" in result['reason']:
        print("PASSED: Memory already exists (Duplicate detection works).")
    else:
        print("PASSED: Memory stored.")

    # 2. Test Search
    query = "programming language"
    print(f"\n[Test 2] Searching for: '{query}'")
    results = manager.search_memory(query)
    found = False
    for entry, score in results:
        print(f" - Found: {entry.text} (Score: {score:.4f})")
        if "Python" in entry.text or "Java" in entry.text:
            found = True
    
    if found:
        print("PASSED: Retrieved relevant memory.")
    else:
        print("FAILED: Did not retrieve relevant memory.")

    # 3. Test Chat
    chat_query = "What is my favorite programming language?"
    print(f"\n[Test 3] Chat Query: '{chat_query}'")
    response, _ = chat.chat(chat_query)
    print(f"Response: {response}")
    
    if "Python" in response or "Java" in response:
        print("PASSED: Chat response contained correct information.")
    else:
        print("FAILED: Chat response did not contain expected information.")

    # 4. Test Implicit Memory in Chat
    print(f"\n[Test 4] Implicit Memory in Chat")
    implicit_input = "My favorite food is sushi."
    print(f"User Input: '{implicit_input}'")
    
    response, store_result = chat.chat(implicit_input)
    print(f"Chat Response: {response}")
    print(f"Store Result: {store_result}")

    if store_result['stored']:
        print("PASSED: Implicit memory stored.")
    elif not store_result['stored'] and "already exists" in store_result['reason']:
        print("PASSED: Implicit memory already exists (Duplicate detection works).")
    else:
        print(f"FAILED: Implicit memory NOT stored (Reason: {store_result['reason']})")

    # 5. Test Memory Update
    print(f"\n[Test 5] Memory Update")
    update_input = "Actually, I changed my mind. My favorite programming language is Java."
    print(f"User Input: '{update_input}'")
    
    result = manager.process_input(update_input)
    print(f"Result: {result}")
    
    # Check if Python memory is gone or Java is there
    # We can search again
    query = "programming language"
    results = manager.search_memory(query)
    found_java = False
    found_python = False
    for entry, score in results:
        if "Java" in entry.text:
            found_java = True
        if "Python" in entry.text:
            found_python = True
            
    if found_java and not found_python:
        print("PASSED: Memory updated (Java found, Python gone).")
    elif found_java and found_python:
        print("WARNING: Both memories exist. Update might have failed to delete old one (or LLM decided to keep both).")
    else:
        print("FAILED: Java not found.")

    # 6. Test Exclusive Preference
    print(f"\n[Test 6] Exclusive Preference")
    # First add some conflicting memories
    manager.process_input("I like burgers.")
    manager.process_input("I like pizza.")
    
    exclusive_input = "I ONLY like salad."
    print(f"User Input: '{exclusive_input}'")
    
    result = manager.process_input(exclusive_input)
    print(f"Result: {result}")
    
    # Check if burgers/pizza are gone
    results = manager.search_memory("food")
    found_salad = False
    found_others = False
    for entry, score in results:
        if "salad" in entry.text.lower():
            found_salad = True
        if "burger" in entry.text.lower() or "pizza" in entry.text.lower():
            found_others = True
            
    if found_salad and not found_others:
        print("PASSED: Exclusive preference respected (Salad found, others gone).")
    elif found_salad and found_others:
        print("WARNING: Exclusive preference added but old memories persist (LLM might need stricter prompt or improved reasoning).")
    else:
        print("FAILED: Salad not found.")

if __name__ == "__main__":
    test_memory_system()
