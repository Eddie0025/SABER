import os
import sys
from saber.llm_engine import LLMEngine

TEST_CASES = [
    # ==========================================================
    # NETWORK SECURITY
    # ==========================================================
    {
        "id": 1,
        "category": "Network Security",
        "question": "A host can ping another server but cannot access HTTPS websites. Walk through your troubleshooting process."
    },
    {
        "id": 2,
        "category": "Network Security",
        "question": "Explain the difference between a stateful firewall and a stateless firewall."
    },
    {
        "id": 3,
        "category": "Network Security",
        "question": "How would you detect DNS tunneling in enterprise network traffic?"
    },
    # ==========================================================
    # WEB SECURITY
    # ==========================================================
    {
        "id": 4,
        "category": "Web Security",
        "question": "Differentiate SQL Injection, Command Injection, and LDAP Injection."
    },
    {
        "id": 5,
        "category": "Web Security",
        "question": "Explain the difference between reflected, stored, and DOM-based XSS."
    },
    {
        "id": 6,
        "category": "Web Security",
        "question": "A website is vulnerable to CSRF. Explain how you would mitigate it."
    },
    # ==========================================================
    # CRYPTOGRAPHY
    # ==========================================================
    {
        "id": 7,
        "category": "Cryptography",
        "question": "Why should passwords be hashed with bcrypt or Argon2 instead of SHA-256?"
    },
    {
        "id": 8,
        "category": "Cryptography",
        "question": "Explain symmetric vs asymmetric encryption with practical use cases."
    },
    {
        "id": 9,
        "category": "Cryptography",
        "question": "Why is reusing an IV in AES-CBC considered insecure?"
    },
    # ==========================================================
    # MALWARE
    # ==========================================================
    {
        "id": 10,
        "category": "Malware",
        "question": "Differentiate a virus, worm, trojan, ransomware, and rootkit."
    },
    {
        "id": 11,
        "category": "Malware",
        "question": "A workstation suddenly begins encrypting files and making thousands of SMB connections. What type of attack do you suspect, and what should be your first response?"
    },
    {
        "id": 12,
        "category": "Malware",
        "question": "Explain the stages of ransomware execution from initial infection to encryption."
    },
    # ==========================================================
    # INCIDENT RESPONSE
    # ==========================================================
    {
        "id": 13,
        "category": "Incident Response",
        "question": "A domain administrator account is suspected to be compromised. Walk through your incident response process."
    },
    {
        "id": 14,
        "category": "Incident Response",
        "question": "Explain the six phases of incident response."
    },
    {
        "id": 15,
        "category": "Incident Response",
        "question": "When should an infected endpoint be isolated from the network?"
    },
    # ==========================================================
    # AUTHENTICATION
    # ==========================================================
    {
        "id": 16,
        "category": "Authentication",
        "question": "Explain Kerberos authentication step-by-step."
    },
    {
        "id": 17,
        "category": "Authentication",
        "question": "Differentiate Pass-the-Hash and Pass-the-Ticket attacks."
    },
    {
        "id": 18,
        "category": "Authentication",
        "question": "Explain why Multi-Factor Authentication greatly reduces account compromise."
    },
    # ==========================================================
    # CLOUD SECURITY
    # ==========================================================
    {
        "id": 19,
        "category": "Cloud",
        "question": "An S3 bucket is accidentally exposed publicly. What risks exist, and what steps should be taken?"
    },
    {
        "id": 20,
        "category": "Cloud",
        "question": "Explain the Shared Responsibility Model in cloud security."
    },
    # ==========================================================
    # FORENSICS
    # ==========================================================
    {
        "id": 21,
        "category": "Digital Forensics",
        "question": "Explain the importance of maintaining chain of custody during an investigation."
    },
    {
        "id": 22,
        "category": "Digital Forensics",
        "question": "A Linux authentication log shows repeated failed SSH logins followed by a successful login from the same IP. What does this suggest?"
    },
    # ==========================================================
    # SECURE CODING
    # ==========================================================
    {
        "id": 23,
        "category": "Secure Coding",
        "question": """Find the vulnerability.

query = "SELECT * FROM users WHERE username='" + user + "'"
"""
    },
    {
        "id": 24,
        "category": "Secure Coding",
        "question": "Why is using eval() on untrusted user input dangerous?"
    },
    # ==========================================================
    # THREAT MODELING
    # ==========================================================
    {
        "id": 25,
        "category": "Threat Modeling",
        "question": "Explain the STRIDE threat modeling framework."
    },
    {
        "id": 26,
        "category": "Threat Modeling",
        "question": "How would you perform a threat model for an online banking application?"
    },
    # ==========================================================
    # RED TEAM / BLUE TEAM
    # ==========================================================
    {
        "id": 27,
        "category": "Operations",
        "question": "Differentiate penetration testing, vulnerability assessment, and red teaming."
    },
    {
        "id": 28,
        "category": "Operations",
        "question": "Explain the MITRE ATT&CK framework and why defenders use it."
    },
    # ==========================================================
    # SECURITY ENGINEERING
    # ==========================================================
    {
        "id": 29,
        "category": "Security Engineering",
        "question": "Design a Zero Trust architecture for a medium-sized enterprise."
    },
    {
        "id": 30,
        "category": "Security Engineering",
        "question": "A company's VPN credentials have been leaked online. Describe your immediate response and long-term remediation plan."
    }
]

def main():
    model_path = "models/cyber_v2"
    
    print("=========================================================")
    print(f" SABER Cybersecurity Evaluation Suite (30 Cases)")
    print(f" Loading Model: {model_path}")
    print("=========================================================")
    
    if not os.path.exists(model_path):
        print(f"Error: Cybersecurity specialist model path '{model_path}' not found.")
        sys.exit(1)
        
    try:
        engine = LLMEngine(model_path)
        engine.__enter__()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print("\nModel loaded successfully! Beginning evaluation...\n")
    
    for case in TEST_CASES:
        print(f"---------------------------------------------------------")
        print(f"CASE [{case['id']}/30] | Category: {case['category']}")
        print(f"Q: {case['question']}")
        print("---------------------------------------------------------")
        
        history = [
            {
                "role": "system",
                "content": (
                    "You are a cybersecurity specialist with expertise in MITRE ATT&CK, "
                    "incident response, threat intelligence, vulnerability analysis, "
                    "and digital forensics. Map threats to specific techniques and "
                    "provide structured analysis. Think through your reasoning "
                    "step by step before providing your final answer."
                )
            }
        ]
        
        try:
            ans = engine.generate_with_history(history, new_user_message=case['question'])
            print(f"SABER: {ans}\n\n")
        except Exception as e:
            print(f"SABER: [FAILED TO GENERATE] {e}\n\n")
            
    engine.__exit__(None, None, None)
    print("=========================================================")
    print(" Evaluation Complete! Model unloaded from VRAM.")
    print("=========================================================")

if __name__ == "__main__":
    main()
