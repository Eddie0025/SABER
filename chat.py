#!/usr/bin/env python3
"""
SABER — Interactive CLI Chat Interface

Usage:
    python3 chat.py
    
Commands:
    /clear   - Reset conversation history
    /domain  - Show last routed domain
    /quit    - Exit SABER
"""

import sys
import logging
from saber.specialist import SpecialistEngine
from saber.orchestrator import Orchestrator
from saber.context import SessionContext

# ─── Logging Setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/saber.log"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("SABER_Chat")

# ─── ANSI Colors ──────────────────────────────────────────────────
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

BANNER = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███████╗ █████╗ ██████╗ ███████╗██████╗                    ║
║   ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗                   ║
║   ███████╗███████║██████╔╝█████╗  ██████╔╝                   ║
║   ╚════██║██╔══██║██╔══██╗██╔══╝  ██╔══██╗                   ║
║   ███████║██║  ██║██████╔╝███████╗██║  ██║                   ║
║   ╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝                   ║
║                                                              ║
║   Specialist Agent-Based Expert Reasoning                    ║
║   Qwen2.5-7B-Instruct + 9 DoRA Specialists                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{RESET}

{DIM}Commands: /clear (reset), /domain (show routing), /quit (exit){RESET}
"""


def main():
    import os
    os.makedirs("logs", exist_ok=True)

    print(BANNER)
    print(f"{YELLOW}Initializing SABER engine...{RESET}")

    # Initialize engine
    engine = SpecialistEngine()
    engine.load_base_model()

    orchestrator = Orchestrator(engine)
    context = SessionContext(max_turns=20)

    print(f"{GREEN}SABER is ready. Type your query below.{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}{CYAN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Goodbye!{RESET}")
            engine.shutdown()
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                print(f"\n{YELLOW}Shutting down SABER...{RESET}")
                engine.shutdown()
                break
            elif cmd == "/clear":
                context.clear()
                print(f"{DIM}Conversation cleared.{RESET}\n")
                continue
            elif cmd == "/domain":
                last = context.get_metadata("last_domain")
                print(f"{DIM}Last routed domain: {last or 'None'}{RESET}\n")
                continue
            else:
                print(f"{DIM}Unknown command: {user_input}{RESET}\n")
                continue

        # Process the query
        try:
            response = orchestrator.process(user_input, context)
            print(f"\n{BOLD}{GREEN}SABER:{RESET} {response}\n")
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"\n{RED}Error: {e}{RESET}\n")


if __name__ == "__main__":
    main()
