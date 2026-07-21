import os
import sys
from saber.llm_engine import LLMEngine

TEST_CASES = [

# ==========================================================
# THREAT MODELING (1-6)
# ==========================================================

{
    "id": 1,
    "category": "Threat Modeling",
    "question": "Design a threat model for a nationwide digital banking platform. Identify the primary assets, threat actors, attack surfaces, and highest-risk attack scenarios."
},

{
    "id": 2,
    "category": "Threat Modeling",
    "question": "An AI-powered hospital platform stores patient records, medical images, and live telemetry. Construct a threat model and prioritize the risks."
},

{
    "id": 3,
    "category": "Threat Modeling",
    "question": "Your company is migrating from an on-premise monolith to Kubernetes. Explain how the threat landscape changes."
},

{
    "id": 4,
    "category": "Threat Modeling",
    "question": "Identify the security risks introduced by exposing an internal REST API to third-party developers."
},

{
    "id": 5,
    "category": "Threat Modeling",
    "question": "Threat model an autonomous drone fleet used for package delivery."
},

{
    "id": 6,
    "category": "Threat Modeling",
    "question": "Threat model a distributed AI agent system where multiple autonomous agents communicate over the public Internet."
},

# ==========================================================
# INCIDENT RESPONSE (7-12)
# ==========================================================

{
    "id": 7,
    "category": "Incident Response",
    "question": "Multiple employees report that all shared files suddenly have a '.locked' extension. Describe your first ten incident response actions in order."
},

{
    "id": 8,
    "category": "Incident Response",
    "question": "A domain administrator account begins authenticating simultaneously from India, Germany, and Brazil. Explain your investigation process."
},

{
    "id": 9,
    "category": "Incident Response",
    "question": "A production database is leaking customer records. Explain how you would contain the incident while preserving forensic evidence."
},

{
    "id": 10,
    "category": "Incident Response",
    "question": "A Kubernetes cluster is suspected to have been compromised. Describe your investigation methodology."
},

{
    "id": 11,
    "category": "Incident Response",
    "question": "An employee accidentally uploads confidential source code to a public GitHub repository. Explain the incident response process."
},

{
    "id": 12,
    "category": "Incident Response",
    "question": "A company discovers malware communicating with an unknown command-and-control server. Outline your response from detection to eradication."
},

# ==========================================================
# DETECTION ENGINEERING (13-18)
# ==========================================================

{
    "id": 13,
    "category": "Detection Engineering",
    "question": "Describe how you would detect credential stuffing attacks without relying solely on failed login counts."
},

{
    "id": 14,
    "category": "Detection Engineering",
    "question": "How would you distinguish legitimate PowerShell administration from malicious PowerShell activity?"
},

{
    "id": 15,
    "category": "Detection Engineering",
    "question": "Design a detection strategy for lateral movement within an Active Directory environment."
},

{
    "id": 16,
    "category": "Detection Engineering",
    "question": "How would you detect data exfiltration over HTTPS without decrypting all traffic?"
},

{
    "id": 17,
    "category": "Detection Engineering",
    "question": "Design detections for insider threats involving privileged users."
},

{
    "id": 18,
    "category": "Detection Engineering",
    "question": "How would you detect ransomware before file encryption begins?"
},

# ==========================================================
# SECURITY ARCHITECTURE (19-24)
# ==========================================================

{
    "id": 19,
    "category": "Architecture",
    "question": "Explain how Zero Trust differs from a traditional perimeter-based security model."
},

{
    "id": 20,
    "category": "Architecture",
    "question": "Design a secure authentication architecture for a global SaaS platform serving 100 million users."
},

{
    "id": 21,
    "category": "Architecture",
    "question": "Explain why network segmentation limits attacker movement even after initial compromise."
},

{
    "id": 22,
    "category": "Architecture",
    "question": "Describe the security architecture for protecting software supply chains."
},

{
    "id": 23,
    "category": "Architecture",
    "question": "Design a secure secrets management strategy for thousands of microservices."
},

{
    "id": 24,
    "category": "Architecture",
    "question": "Explain how hardware-backed key storage changes an organization's security posture."
},

# ==========================================================
# CRYPTOGRAPHY & SECURITY REASONING (25-30)
# ==========================================================

{
    "id": 25,
    "category": "Cryptography",
    "question": "Explain why simply increasing encryption key length does not automatically improve overall system security."
},

{
    "id": 26,
    "category": "Cryptography",
    "question": "Compare symmetric encryption, asymmetric encryption, hashing, and digital signatures. Explain when each should be used."
},

{
    "id": 27,
    "category": "Security Reasoning",
    "question": "A company requires multi-factor authentication but users complain about usability. Explain how you would balance security and user experience."
},

{
    "id": 28,
    "category": "Security Reasoning",
    "question": "Explain why security through obscurity should never be the primary defense mechanism."
},

{
    "id": 29,
    "category": "Security Reasoning",
    "question": "A vulnerability scanner reports 10,000 findings across an enterprise. Explain how you would prioritize remediation rather than fixing vulnerabilities by CVSS score alone."
},

{
    "id": 30,
    "category": "Integrated Security",
    "question": "Design a defense-in-depth strategy for protecting a nationwide AI infrastructure from ransomware, insider threats, supply-chain attacks, and nation-state adversaries."
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
