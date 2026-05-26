"""
Stack Trail Chatbot — Terminal chatbot powered by the Anthropic API.

SETUP:
    pip install anthropic rich
    export ANTHROPIC_API_KEY="your-api-key"

USAGE:
    python chatbot.py

COMMANDS:
    /clear    Reset conversation history
    /history  Show all past messages
    /exit     Quit
"""

import os
import sys
from anthropic import Anthropic, APIError, APIConnectionError, APIStatusError
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text


SYSTEM_PROMPT = """You are a helpful, knowledgeable, and friendly AI assistant. \
Provide clear, accurate, and concise responses."""

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192

console = Console()
client = Anthropic()
history: list[dict] = []


def welcome_banner() -> None:
    title = Text("Stack Trail Chatbot", style="bold cyan")
    subtitle = Text.assemble(
        ("Model: ", "dim"), (MODEL, "cyan"),
        ("  •  ", "dim"),
        ("/clear", "bold yellow"), ("  ", "dim"),
        ("/history", "bold yellow"), ("  ", "dim"),
        ("/exit", "bold yellow"),
    )
    content = Text.assemble(title, "\n", subtitle)
    console.print(Panel(content, border_style="cyan", padding=(1, 2)))
    console.print()


def show_help() -> None:
    console.print("[bold]Commands:[/bold]")
    console.print("  [bold yellow]/help[/bold yellow]     show this message")
    console.print("  [bold yellow]/clear[/bold yellow]    reset conversation history")
    console.print("  [bold yellow]/history[/bold yellow]  show past messages")
    console.print("  [bold yellow]/exit[/bold yellow]     quit")
    console.print()


def show_history() -> None:
    if not history:
        console.print("[dim]No conversation history yet.[/dim]\n")
        return
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            console.print(f"[bold blue]You:[/bold blue] {content}")
        else:
            console.print("[bold green]Assistant:[/bold green]")
            console.print(Markdown(content))
        console.print()


def stream_response(user_input: str) -> None:
    history.append({"role": "user", "content": user_input})
    full_response = ""
    usage = None

    try:
        with Live(
            Spinner("dots", text=" [dim]Thinking...[/dim]"),
            console=console,
            refresh_per_second=15,
            vertical_overflow="visible",
        ) as live:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=history,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    live.update(
                        Panel(
                            Markdown(full_response),
                            title="[bold green]Assistant[/bold green]",
                            border_style="green",
                            padding=(0, 1),
                        )
                    )
                usage = stream.get_final_message().usage

    except APIConnectionError:
        console.print("\n[bold red]Connection error.[/bold red] Check your network.\n")
        history.pop()
        return
    except APIStatusError as e:
        console.print(f"\n[bold red]API error {e.status_code}:[/bold red] {e.message}\n")
        history.pop()
        return
    except APIError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        history.pop()
        return

    if full_response:
        history.append({"role": "assistant", "content": full_response})
        if usage:
            console.print(f"[dim]tokens: {usage.input_tokens} in / {usage.output_tokens} out[/dim]")
    console.print()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY environment variable not set.")
        console.print("Run: [cyan]export ANTHROPIC_API_KEY=your-key-here[/cyan]")
        sys.exit(1)

    welcome_banner()

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            console.print("[dim]Goodbye![/dim]")
            break
        elif user_input.lower() == "/clear":
            history.clear()
            console.print("[dim]History cleared.[/dim]\n")
            continue
        elif user_input.lower() == "/history":
            show_history()
            continue
        elif user_input.lower() == "/help":
            show_help()
            continue
        elif user_input.startswith("/"):
            console.print("[yellow]Unknown command.[/yellow] Try /clear, /history, or /exit.\n")
            continue

        stream_response(user_input)


if __name__ == "__main__":
    main()
