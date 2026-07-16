# -*- coding: utf-8 -*-
import json
import time
from saber.llm_engine import LLMEngine

QUESTIONS = [
    {
        "id": "CYBER-001",
        "title": "Incident Response",
        "text": "A Windows Server 2022 domain controller begins making encrypted outbound connections to an IP address in a foreign country every 12 minutes.\n\nSecurity logs show:\n- A PowerShell script executed from a user's Downloads folder.\n- A new local administrator account was created.\n- Multiple ZIP archives containing HR records were created shortly before the outbound connections began.\n- EDR did not detect malware.\n\nAs a cybersecurity analyst:\n1. Identify the likely attack stages.\n2. Map the activity to MITRE ATT&CK techniques.\n3. Assess severity and business impact.\n4. Identify attacker objectives.\n5. Recommend immediate containment actions.\n6. List forensic evidence that should be collected.\n7. Recommend long-term remediation measures.\n\nProvide technical reasoning for every conclusion."
    },
    {
        "id": "CYBER-002",
        "title": "Risk Prioritization",
        "text": "A recently disclosed vulnerability has CVSS: 9.8. The vulnerability allows Remote Code Execution.\nHowever:\n- Authentication is required.\n- The attacker must already possess a privileged role.\n- The endpoint is only accessible through an internal VPN.\n- A Web Application Firewall protects the application.\n\nAs a cybersecurity analyst:\n1. Explain whether the CVSS score reflects the real-world risk.\n2. Identify all exploitation prerequisites.\n3. Assess attack likelihood.\n4. Assess business impact.\n5. Recommend patch prioritization.\n6. Recommend compensating controls.\n7. Explain your reasoning for every recommendation."
    },
    {
        "id": "CYBER-003",
        "title": "Cloud Security",
        "text": "A healthcare company wants to migrate patient records to AWS.\n\nRequirements:\n- HIPAA compliance\n- Multi-region availability\n- Protection against ransomware\n- Insider threat monitoring\n- Least privilege access\n- Full audit logging\n\nDesign a secure architecture.\n\nInclude:\n1. Network design\n2. IAM strategy\n3. Encryption strategy\n4. Backup strategy\n5. Monitoring strategy\n6. Incident response considerations\n\nJustify every architectural decision."
    },
    {
        "id": "CYBER-004",
        "title": "Threat Hunting",
        "text": "You are given the following indicators:\n- Repeated DNS requests to random subdomains\n- Outbound HTTPS traffic to a newly registered domain\n- PowerShell spawning cmd.exe\n- LSASS memory access detected\n\nAs a threat hunter:\n1. Explain what attack activity may be occurring.\n2. Identify likely attacker goals.\n3. Map activity to MITRE ATT&CK.\n4. Recommend hunting queries.\n5. Recommend detection opportunities.\n6. Recommend containment actions."
    },
    {
        "id": "CYBER-005",
        "title": "Detection Engineering",
        "text": "A ransomware group frequently performs:\n- PowerShell execution\n- Credential dumping\n- PsExec lateral movement\n- Mass file encryption\n\nAs a detection engineer:\n1. Design detection logic.\n2. Identify required log sources.\n3. Recommend SIEM alerts.\n4. Recommend Sigma rules.\n5. Explain how false positives should be handled."
    },
    {
        "id": "CYBER-006",
        "title": "Security Architecture",
        "text": "A company has:\n- 2,500 employees\n- Hybrid cloud environment\n- Active Directory\n- AWS workloads\n- Remote workforce\n\nDesign a Zero Trust architecture.\n\nExplain:\n1. Identity controls\n2. Network segmentation\n3. Device trust\n4. Privileged access management\n5. Continuous verification\n6. Logging and monitoring"
    },
    {
        "id": "CYBER-007",
        "title": "Adversarial Reasoning Test",
        "text": "A vulnerability scanner reports:\nCVSS 10.0\nCritical\n\nThe vulnerable system:\n- Is air-gapped\n- Has no external connectivity\n- Requires physical access\n- Processes non-sensitive data\n\nManagement wants emergency remediation.\n\nAs a security analyst:\n1. Assess actual risk.\n2. Explain why CVSS may or may not be sufficient.\n3. Recommend remediation priority.\n4. Explain business tradeoffs.\n5. Provide a final recommendation."
    },
    {
        "id": "CYBER-008",
        "title": "Red vs Blue Thinking",
        "text": "Assume an attacker has compromised a standard employee account.\n\nThe environment contains:\n- Active Directory\n- Azure AD\n- VPN access\n- File shares\n- Endpoint protection\n\n1. Describe likely attacker actions.\n2. Describe likely privilege escalation paths.\n3. Describe lateral movement possibilities.\n4. Recommend detection opportunities.\n5. Recommend defensive measures."
    },
    {
        "id": "CYBER-009",
        "title": "Executive Communication",
        "text": "A ransomware attack encrypted 30% of company file servers.\n\nThe CEO wants a briefing.\n\nProvide:\n1. Executive summary\n2. Technical summary\n3. Business impact\n4. Immediate actions\n5. Recovery strategy\n6. Long-term improvements\n\nExplain complex technical issues in business language."
    },
    {
        "id": "CYBER-010",
        "title": "SABER Killer Question",
        "text": "A hospital suffers a ransomware attack.\n\nEffects:\n- Patient scheduling system unavailable\n- MRI machines disconnected from the network\n- Staff using personal devices for communication\n- Backup servers appear unaffected\n- Suspicious outbound traffic observed before encryption\n\nAs a cybersecurity specialist:\n1. Analyze the attack.\n2. Identify likely attacker objectives.\n3. Assess patient safety risks.\n4. Assess operational risks.\n5. Assess legal and compliance risks.\n6. Recommend containment actions.\n7. Recommend recovery actions.\n8. Recommend communication strategy.\n\nProvide technical reasoning for every conclusion."
    }
]

def main():
    model_path = "models/cyber_v1"
    output_file = "/Users/adityavir/.gemini/antigravity-ide/brain/f0e0302e-3bd0-4295-aa04-85662d5a792a/cyber_v1_results.md"
    
    with open(output_file, "w") as f:
        f.write("# Cyber-v1 Direct Benchmark Results\n\n")
        f.write(f"Evaluating {len(QUESTIONS)} questions directly against `{model_path}` (no SABER orchestrator/verifier).\n\n")

    print(f"Loading {model_path}...")
    try:
        with LLMEngine(model_path) as engine:
            for i, q in enumerate(QUESTIONS, 1):
                print(f"[{i}/{len(QUESTIONS)}] Testing {q['id']}...")
                start = time.time()
                
                system_prompt = "You are a senior cybersecurity expert. Provide a detailed, accurate, and structured answer."
                answer = engine.generate(q['text'], system_prompt=system_prompt).strip()
                
                latency = time.time() - start
                print(f"  Done in {latency:.1f}s")
                print("\n" + "="*50)
                print(f"Question: {q['id']}")
                print("="*50)
                print(answer)
                print("="*50 + "\n", flush=True)
                
                # Append to artifact
                with open(output_file, "a") as f:
                    f.write(f"## {q['id']} — {q['title']}\n")
                    f.write(f"**Latency:** {latency:.2f}s\n\n")
                    f.write("### Question\n```text\n")
                    f.write(q['text'])
                    f.write("\n```\n\n### Cyber-v1 Answer\n")
                    f.write(answer)
                    f.write("\n\n---\n\n")
                    
    except Exception as e:
        print(f"Failed to load/run engine: {e}")
        
    print("Benchmark complete!")

if __name__ == "__main__":
    main()
