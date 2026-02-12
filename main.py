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
        self.chat_service = ChatService(memory_manager=self.memory_manager)
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

    def train_retriever_flow(self):
        """Trains the contrastive retriever on stored memories."""
        console.print("[bold magenta]Train Contrastive Retriever[/bold magenta]\n")

        # Check memory count
        mem_count = self.memory_manager.storage_service.get_memory_count()
        console.print(f"Stored memories: [cyan]{mem_count}[/cyan]")

        if mem_count < 5:
            console.print("[red]Need at least 5 memories to train. Add more memories first.[/red]")
            Prompt.ask("\nPress Enter to return")
            return

        trainer = self.memory_manager.contrastive_trainer
        if trainer is None or trainer.model is None:
            console.print("[red]sentence-transformers not installed. Run: pip install sentence-transformers[/red]")
            Prompt.ask("\nPress Enter to return")
            return

        is_retrain = trainer.is_trained()
        if is_retrain:
            console.print("[yellow]Existing model found — will warm-start from previous checkpoint.[/yellow]")
        else:
            console.print("[green]First training — starting from base MiniLM model.[/green]")

        # Step 1: Generate triplets
        console.print("\n[bold yellow]Step 1: Generating training triplets...[/bold yellow]")
        from src.services.triplet_service import TripletService
        triplet_service = TripletService(self.memory_manager.llm_service, self.memory_manager.embedding_service)

        memories = self.memory_manager.storage_service.get_all_memories()
        with console.status(f"[bold blue]Asking LLM to generate queries for {len(memories)} memories...[/bold blue]"):
            triplets = triplet_service.generate_triplets(memories, num_queries_per_memory=2)

        if len(triplets) < 3:
            console.print("[red]Not enough triplets generated. Try adding more diverse memories.[/red]")
            Prompt.ask("\nPress Enter to return")
            return

        console.print(f"Generated [cyan]{len(triplets)}[/cyan] training triplets.")

        # Show a few examples
        table = Table(title="Sample Triplets", show_lines=True)
        table.add_column("Query", style="green", max_width=30)
        table.add_column("Positive", style="cyan", max_width=30)
        table.add_column("Negative", style="red", max_width=30)
        for t in triplets[:3]:
            table.add_row(t["query"], t["positive"], t["negative"])
        console.print(table)

        # Step 2: Train
        console.print("\n[bold yellow]Step 2: Training contrastive model...[/bold yellow]")
        with console.status("[bold blue]Fine-tuning MiniLM with TripletLoss...[/bold blue]"):
            metrics = trainer.train(triplets, epochs=3, batch_size=16)

        if "error" in metrics:
            console.print(f"[red]Training failed: {metrics['error']}[/red]")
            Prompt.ask("\nPress Enter to return")
            return

        console.print(Panel(
            f"Model version: [cyan]v{metrics['model_version']}[/cyan]\n"
            f"Training time: [cyan]{metrics['duration_seconds']}s[/cyan]\n"
            f"Triplets used: [cyan]{metrics['num_triplets']}[/cyan]\n"
            f"Saved to: [dim]{metrics['model_path']}[/dim]",
            title="Training Complete", border_style="green"
        ))

        # Step 3: Re-embed all memories
        console.print("\n[bold yellow]Step 3: Re-embedding all memories...[/bold yellow]")
        with console.status("[bold blue]Encoding memories with fine-tuned model...[/bold blue]"):
            count = trainer.reembed_all(self.memory_manager.storage_service)

        console.print(f"[green]Re-embedded {count} memories with the fine-tuned model.[/green]")
        console.print("\n[bold green]Retriever is now active! Future searches will use the learned model.[/bold green]")

        Prompt.ask("\nPress Enter to return")

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
            menu_table.add_row("5", "Train Retriever (Contrastive Learning)")
            menu_table.add_row("6", "Exit")
            console.print(menu_table)
            
            choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5", "6"], default="1")

            if choice == "1":
                self.add_memory_flow()
            elif choice == "2":
                self.search_memory_flow()
            elif choice == "3":
                self.chat_flow()
            elif choice == "4":
                self.voice_chat_flow()
            elif choice == "5":
                self.train_retriever_flow()
            elif choice == "6":
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
