from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import print
from rich.layout import Layout
from src.core.memory_manager import MemoryManager
from src.core.chat_service import ChatService
from src.services.stt_service import STTService
from src.services.wakeword_service import WakeWordService
import time

console = Console()

class MemoryApp:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.chat_service = ChatService()
        self.stt_service = STTService()
        self.wakeword_service = WakeWordService()

    def display_header(self):
        console.clear()
        console.print(Panel.fit("[bold cyan]Quantum Memory Layer Demo[/bold cyan]", border_style="blue"))
        console.print("[dim]Powered by Google Gemini & SQLite[/dim]\n")

    def add_memory_flow(self):
        console.print("[bold green]Add New Memory[/bold green]")
        text = Prompt.ask("Enter the information")
        
        with console.status("[bold blue]Consulting with LLM...[/bold blue]"):
            result = self.memory_manager.process_input(text)
            time.sleep(0.5) # purely for effect

        if result['stored']:
            console.print(Panel(f"[bold green]Memory Stored![/bold green]\n\nReason: {result['reason']}", title="Success", border_style="green"))
        else:
            console.print(Panel(f"[bold yellow]Memory Ignored[/bold yellow]\n\nReason: {result['reason']}", title="Skipped", border_style="yellow"))
        
        Prompt.ask("\nPress Enter to return")

    def search_memory_flow(self):
        console.print("[bold magenta]Search Memories[/bold magenta]")
        query = Prompt.ask("Enter search query")

        with console.status("[bold magenta]Searching...[/bold magenta]"):
            results = self.memory_manager.search_memory(query)
            time.sleep(0.3)

        if not results:
            console.print("[red]No memories found.[/red]")
        else:
            table = Table(title=f"Search Results for '{query}'")
            table.add_column("Score", style="cyan", no_wrap=True)
            table.add_column("Memory", style="white")
            table.add_column("Created At", style="dim")

            for entry, score in results:
                table.add_row(f"{score:.4f}", entry.text, entry.created_at.strftime("%Y-%m-%d %H:%M"))
            
            console.print(table)
        
        Prompt.ask("\nPress Enter to return")

    def voice_chat_flow(self):
        console.clear()
        console.print(Panel.fit("Voice Chat Mode (Wake Word: 'Jarvis' or 'Computer')", border_style="green"))
        console.print("[dim]Say 'exit' or 'quit' to stop.[/dim]\n")
        
        if not self.wakeword_service.porcupine:
             console.print("[yellow]Wake Word detection not available (check AccessKey). Falling back to direct listen.[/yellow]")

        while True:
            # 1. Wait for Wake Word (if available)
            if self.wakeword_service.porcupine:
                console.print("[dim]Waiting for wake word...[/dim]")
                if not self.wakeword_service.listen_for_wake_word():
                     break # Stop if error or interrupted

            # 2. Capture Audio
            user_input = self.stt_service.listen_and_transcribe()
            
            if not user_input:
                console.print("[dim]No speech detected.[/dim]")
                continue
                
            console.print(f"[bold green]You said:[/bold green] {user_input}")
            
            if user_input.lower() in ['exit', 'quit', 'stop']:
                break
            
            # 3. Process Chat (Same logic as text chat)
            with console.status("[bold blue]Thinking...[/bold blue]"):
                response, store_result = self.chat_service.chat(user_input)
            
            console.print(Panel(response, title="Gemini", border_style="blue"))

            if store_result['stored']:
                console.print(f"[dim italic green]Memory stored: {store_result['reason']}[/dim italic green]")
            elif not store_result['stored'] and "already exists" in store_result['reason']:
                console.print(f"[dim italic yellow]Memory already exists.[/dim italic yellow]")
                
            # Optional: Add TTS here if requested later
            console.print("\n")

    def chat_flow(self):
        console.print("[bold cyan]Chat with Memory (Type 'exit' to quit chat)[/bold cyan]")
        
        while True:
            user_input = Prompt.ask("[bold green]You[/bold green]")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            with console.status("[bold blue]Thinking...[/bold blue]"):
                response, store_result = self.chat_service.chat(user_input)
            
            console.print(Panel(response, title="Gemini", border_style="blue"))

            if store_result['stored']:
                console.print(f"[dim italic green]Memory stored: {store_result['reason']}[/dim italic green]")

    def run(self):
        while True:
            self.display_header()
            menu_table = Table(show_header=False, box=None)
            menu_table.add_column("Option", style="cyan", justify="right")
            menu_table.add_column("Description", style="white")
            menu_table.add_row("1", "Add Memory (LLM Decides)")
            menu_table.add_row("2", "Search Memories")
            menu_table.add_row("3", "Chat with Memory")
            menu_table.add_row("4", "Voice Chat")
            menu_table.add_row("5", "Exit")
            console.print(menu_table)
            
            choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5"], default="1")

            if choice == "1":
                self.add_memory_flow()
            elif choice == "2":
                self.search_memory_flow()
            elif choice == "3":
                self.chat_flow()
            elif choice == "4":
                self.voice_chat_flow()
            elif choice == "5":
                console.print("[bold green]Goodbye![/bold green]")
                break

if __name__ == "__main__":
    try:
        app = MemoryApp()
        app.run()
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        console.print("Please check your .env file.")
    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {e}")
