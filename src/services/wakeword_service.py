import pvporcupine
import pvrecorder
from src.core.config import Config
from rich.console import Console

console = Console()

class WakeWordService:
    def __init__(self, keywords=["jarvis", "computer"]):
        """
        Initializes Porcupine with specified keywords.
        Common built-in keywords: 'alexa', 'americano', 'blueberry', 'bumblebee', 'computer', 'grapefruit', 'grasshopper', 'hey google', 'hey siri', 'jarvis', 'ok google', 'picovoice', 'porcupine', 'terminator'.
        """
        try:
            self.porcupine = None
            Config.validate_wakeword()
            self.access_key = Config.PICOVOICE_ACCESS_KEY
            self.keywords = keywords
            self.porcupine = pvporcupine.create(access_key=self.access_key, keywords=self.keywords)
            self.recorder = pvrecorder.PvRecorder(device_index=-1, frame_length=self.porcupine.frame_length)
        except ValueError as e:
            console.print(f"[red]Wake Word Config Error: {e}[/red]")
        except Exception as e:
            console.print(f"[red]Wake Word Init Error: {e}[/red]")

    def listen_for_wake_word(self):
        """
        Blocks and listens until the wake word is detected.
        Returns True if detected, False if error/interrupted.
        """
        if not self.porcupine:
            return False

        try:
            self.recorder.start()
            console.print(f"[bold cyan]Listening for wake word ({', '.join(self.keywords)})...[/bold cyan]")
            
            while True:
                pcm = self.recorder.read()
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    console.print("[bold green]Wake Word Detected![/bold green]")
                    return True
                    
        except KeyboardInterrupt:
            return False
        except Exception as e:
            console.print(f"[red]Error listening for wake word: {e}[/red]")
            self.recorder.stop()
            return False
        finally:
            if self.recorder.is_recording:
                self.recorder.stop()

    def cleanup(self):
        if self.porcupine:
            self.porcupine.delete()
        if hasattr(self, 'recorder'):
            self.recorder.delete()
