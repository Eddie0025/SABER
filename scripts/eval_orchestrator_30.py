import sys
from saber.llm_engine import LLMEngine

QUESTIONS = [
    "I'm opening a digital bank for freelancers across Asia. Help me build the platform.",
    "My hospital still uses paper records. I want to digitize everything and make it secure.",
    "Can you help me create an AI tutor that teaches engineering students and automatically grades assignments?",
    "I'm building a drone that can inspect power lines without human intervention.",
    "I want to launch an online marketplace like Amazon but only for medical equipment.",
    "We're getting hacked constantly. Can you help redesign our infrastructure?",
    "I have thousands of IoT sensors collecting environmental data every second. I need a system that can process it in real time.",
    "I want an app where doctors can remotely monitor diabetic patients using wearable devices.",
    "Help me build software that predicts stock prices and automatically manages investments.",
    "We're replacing spreadsheets with a proper ERP across five factories.",
    "I need software for a self-driving delivery robot operating inside hospitals.",
    "My startup is building a password manager for enterprises.",
    "I want to build something similar to ChatGPT but only for legal documents.",
    "I need a system that detects fraud in banking transactions before money leaves the account.",
    "Can you help me design software for controlling satellites?",
    "I want to create an app where users can upload videos, livestream, chat, and pay creators.",
    "We're building a nationwide electronic voting system.",
    "My company wants to migrate from a monolith to microservices over the next two years.",
    "We have dozens of legacy applications and management wants 'everything moved to the cloud.'",
    "I want to create an AI research assistant that reads papers, summarizes them, and writes literature reviews.",
    "Our hospital was hit by ransomware last week. We need a long-term recovery and prevention strategy.",
    "We're opening a cryptocurrency exchange and expect millions of users on day one.",
    "My manufacturing plant wants predictive maintenance using machine learning and sensor data.",
    "I'm building software for autonomous warehouse robots that need to avoid collisions and optimize routes.",
    "I want a platform where researchers can securely share confidential medical datasets across universities.",
    "Help me build software for managing an airport, including flights, baggage, security, and passenger services.",
    "We're building a battery-less NFC smart bandage that continuously monitors diabetic wounds and sends data to a mobile app.",
    "My university wants to detect cheating during online exams using AI while keeping student privacy protected.",
    "We're launching a digital insurance company that uses AI to approve claims automatically while preventing fraud.",
    "I want to build a smart city platform that manages traffic lights, emergency services, utilities, environmental sensors, and citizen services from one dashboard."
]

def main():
    model_path = "models/orchestrator_v2"
    
    print("=========================================================")
    print(f" SABER Orchestrator Evaluation Suite v3 (Realistic Queries)")
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
