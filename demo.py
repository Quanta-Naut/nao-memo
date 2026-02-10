import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import print
from src.core.memory_manager import MemoryManager
from src.core.chat_service import ChatService

console = Console()

class DemoApp:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.chat_service = ChatService()

    def run(self):
        console.clear()
        console.print(Panel.fit("[bold cyan]Gemini Memory Layer - Internals Demo[/bold cyan]", border_style="blue"))
        console.print("[dim]This mode shows embedding vectors, search scores, and LLM reasoning.[/dim]\n")

        while True:
            user_input = Prompt.ask("\n[bold green]Enter Query/Message (or 'exit')[/bold green]")
            if user_input.lower() in ['exit', 'quit']:
                break

            # 1. Generate Embedding
            console.print("\n[bold yellow]--- Step 1: Generating Embedding ---[/bold yellow]")
            with console.status("Generating embedding..."):
                embedding = self.memory_manager.embedding_service.get_embedding(user_input)
            
            # Show a snippet of the vector
            vec_preview = f"[{', '.join(f'{x:.4f}' for x in embedding[:5])}, ... ({len(embedding)} dimensions)]"
            console.print(f"Vector: [italic]{vec_preview}[/italic]")

            # 2. Search Memories
            console.print("\n[bold yellow]--- Step 2: Searching Memories (Cosine Similarity) ---[/bold yellow]")
            with console.status("Searching..."):
                results = self.memory_manager.search_memory(user_input, limit=5)
            
            if not results:
                console.print("[red]No memories found in DB.[/red]")
            else:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Similarity Score")
                table.add_column("Memory Text")
                
                context = []
                for entry, score in results:
                    style = "green" if score > 0.6 else "dim"
                    table.add_row(f"{score:.4f}", entry.text, style=style)
                    if score > 0.6:
                        context.append(entry.text)
                
                console.print(table)

            # 3. Decision to Store
            console.print("\n[bold yellow]--- Step 3: Memory Storage Decision ---[/bold yellow]")
            with console.status("Consulting LLM..."):
                store_result = self.memory_manager.process_input(user_input)
            
            if store_result['stored']:
                console.print(f"[bold green]DECISION: SAVE[/bold green]")
                console.print(f"Reason: {store_result['reason']}")
            else:
                console.print(f"[bold dim]DECISION: IGNORE[/bold dim]")
                console.print(f"Reason: {store_result['reason']}")

            # 4. Generate Response
            console.print("\n[bold yellow]--- Step 4: Generating LLM Response ---[/bold yellow]")
            if context:
                console.print(f"[dim]Context provided to LLM:\n" + "\n".join(f"- {c}" for c in context) + "[/dim]")
            else:
                console.print("[dim]No relevant context provided.[/dim]")
                
            with console.status("Thinking..."):
                response = self.chat_service.llm_service.generate_chat_response(user_input, context)
            
            console.print(Panel(response, title="Final Response", border_style="blue"))

if __name__ == "__main__":
    DemoApp().run()
