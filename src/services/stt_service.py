import speech_recognition as sr
from rich.console import Console

console = Console()

class STTService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def listen_and_transcribe(self) -> str:
        """
        Listens to the microphone and converts speech to text using Google STT.
        Returns the transcribed text or None if failed.
        """
        try:
            with sr.Microphone() as source:
                console.print("[bold yellow]Listening... (Speak now)[/bold yellow]")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                console.print("[dim]Processing audio...[/dim]")
                
                # Transcribe
                # Note: This uses the default Google API key provided by the library for testing.
                text = self.recognizer.recognize_google(audio)
                return text
                
        except sr.WaitTimeoutError:
            console.print("[red]No speech detected. Timed out.[/red]")
            return None
        except sr.UnknownValueError:
            console.print("[red]Could not understand audio.[/red]")
            return None
        except sr.RequestError as e:
            console.print(f"[red]Could not request results from Google Speech Recognition service; {e}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Microphone Error: {e}[/red]")
            return None
