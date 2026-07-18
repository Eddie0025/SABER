import sys
from saber.llm_engine import LLMEngine

QUESTIONS = [
    # =========================================================
    # Category 1 – Simple Routing (6)
    # =========================================================
    "Explain the CAP theorem.",
    "Design a secure login API.",
    "Diagnose why this SQL query is slow.",
    "Explain diabetic ketoacidosis.",
    "Design a Kubernetes deployment strategy.",
    "Analyze this malware behavior.",

    # =========================================================
    # Category 2 – Multi-Specialist Routing (6)
    # =========================================================
    "Design a secure hospital management system.",
    "Build an AI-powered medical diagnosis platform.",
    "Design a blockchain-based voting system with security requirements.",
    "Design an autonomous drone navigation system.",
    "Design an online banking backend resistant to cyber attacks.",
    "Build a medical chatbot for hospitals.",

    # =========================================================
    # Category 3 – Task Decomposition (6)
    # =========================================================
    "Build a food delivery platform.",
    "Build an e-commerce website.",
    "Design a ride-sharing application.",
    "Build a university ERP.",
    "Build an IoT smart home system.",
    "Create an AI coding assistant.",

    # =========================================================
    # Category 4 – Dependency Planning (4)
    # =========================================================
    "A database schema hasn't been designed yet. Should API implementation begin?",
    "Can penetration testing occur before deployment?",
    "Should caching be implemented before the application works?",
    "Should monitoring be designed after production?",

    # =========================================================
    # Category 5 – Specialist Selection Edge Cases (4)
    # =========================================================
    "Explain hypertension.",
    "Explain RSA encryption.",
    "Design a scalable authentication platform.",
    "Compare pneumonia and tuberculosis.",

    # =========================================================
    # Category 6 – Planning Quality (4)
    # =========================================================
    "User wants an autonomous hospital robot.",
    "User wants a secure cloud-native fintech platform.",
    "User wants a military drone swarm.",
    "User wants a smart city management platform."
]

def main():
    model_path = "models/orchestrator_v2"
    
    print("=========================================================")
    print(f" SABER Orchestrator Evaluation Suite (30 Cases)")
    print(f" Loading Model: {model_path}")
    print("=========================================================")
    
    try:
        engine = LLMEngine(model_path)
        engine.__enter__()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print("\nModel loaded successfully! Beginning evaluation...\n")
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"---------------------------------------------------------")
        print(f"CASE [{i}/30]")
        print(f"Q: {question}")
        print("---------------------------------------------------------")
        
        # Reset context for each question
        history = [
            {"role": "system", "content": "You are the SABER Orchestrator. Route requests to specialist models, decompose complex tasks, plan dependencies, and synthesize plans."}
        ]
        
        try:
            ans = engine.generate_with_history(history, new_user_message=question)
            print(f"SABER: {ans}\n\n")
        except Exception as e:
            print(f"SABER: [FAILED TO GENERATE] {e}\n\n")
            
    engine.__exit__(None, None, None)
    print("=========================================================")
    print(" Evaluation Complete! Model unloaded from VRAM.")
    print("=========================================================")

if __name__ == "__main__":
    main()
