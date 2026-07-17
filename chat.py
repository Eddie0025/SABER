import argparse
import sys
from saber.llm_engine import LLMEngine

def main():
    parser = argparse.ArgumentParser(description="Chat interactively with a SABER model.")
    parser.add_argument("--model", type=str, default="models/medical_v2", help="Path to the trained model directory or HF model name.")
    parser.add_argument("--system", type=str, default="You are a helpful AI specialist. Answer questions thoroughly and accurately.", help="System prompt to use.")
    args = parser.parse_args()

    print("=========================================================")
    print(f" Loading Model: {args.model}")
    print("=========================================================")
    
    try:
        engine = LLMEngine(args.model)
        # We manually enter context to keep the model permanently in VRAM during the chat
        engine.__enter__()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print("\nModel loaded successfully! Type 'exit' or 'quit' to end the session.")
    print("---------------------------------------------------------")
    
    history = [{"role": "system", "content": args.system}]

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            
            if not user_input.strip():
                continue

            print("\nSABER: Generating...")
            
            # Use generate_with_history to maintain context across multiple messages
            response = engine.generate_with_history(history, new_user_message=user_input)
            
            # Update history for the next turn
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            # Print response, erasing the "Generating..." line
            sys.stdout.write("\033[F\033[K")
            print(f"SABER: {response}")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        engine.__exit__(None, None, None)
        print("Model unloaded from VRAM.")

if __name__ == "__main__":
    main()
