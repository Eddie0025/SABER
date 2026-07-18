# -*- coding: utf-8 -*-
"""saber.training.dataset_loader

Automated Data extraction pipeline for SABER using the HuggingFace `datasets` library.
This script downloads the required public datasets and normalizes them into JSONL files.

Each record is structured as:
    {"id": "...", "text": "<question/prompt>", "label": "<detailed answer>", "domain": "..."}

Quality Rules:
    - Every label must be >= 30 characters (no single-word garbage)
    - No "Unknown" labels
    - Every record must have both text and label

v2 — Hardened datasets with expanded sources:
    Medical: MedMCQA + Medical Flashcards + ChatDoctor + WikiDoc
    Cyber:   MITRE STIX + CyberQA + Trendyol + CyberMetric + Synthetic IR/MITRE
    Science: ScienceQA + SciQ + Hendrycks MATH + ARC-Challenge + CAMEL-AI (physics/chemistry/biology)
"""

import os
import json
import uuid
import hashlib
import requests
# pyrefly: ignore [missing-import]
from datasets import load_dataset


def _jsonl_write(records: list, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    print(f"[dataset_loader] Wrote {len(records)} records to {out_path}")


def _quality_filter(records: list, min_label_len: int = 30) -> list:
    """Remove garbage records: short labels, unknowns, empties, and AI refusals."""
    clean = []
    refusal_phrases = [
        "unknown", "i don't know", "cannot be determined", "not enough information",
        "i cannot answer", "i am an ai", "as an ai language model", "unclear from the context"
    ]
    for rec in records:
        label = rec.get("label", "").strip()
        text = rec.get("text", "").strip()
        # Drop empty or very short labels
        if not text or not label:
            continue
        if len(label) < min_label_len:
            continue
        # Drop records containing refusal phrases
        lower_label = label.lower()
        if any(phrase in lower_label for phrase in refusal_phrases):
            continue
        clean.append(rec)
    return clean


import re
import random

def _split_into_sections(text: str) -> list[str]:
    """Split text into logical sections based on headers, lists, or paragraphs."""
    # Split on headers or bold markers
    parts = re.split(r'\n##+\s+|\n\*\*[^*]+\*\*:\s*', text)
    if len(parts) > 1:
        parts = [p.strip() for p in parts if p.strip()]
    else:
        # Try numbered lists
        parts = re.split(r'\n\d+\.\s+', text)
        if len(parts) > 1:
            parts = [p.strip() for p in parts if p.strip()]
        else:
            # Fall back to paragraphs
            parts = [p.strip() for p in text.split('\n\n') if p.strip()]
            
    # If still one part, split by sentences (rudimentary)
    if len(parts) == 1:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 3:
            mid = len(sentences) // 2
            parts = [" ".join(sentences[:mid]), " ".join(sentences[mid:])]
            
    # Combine small parts, cap max sections
    merged = []
    curr = ""
    for p in parts:
        curr += (" " if curr else "") + p
        if len(curr) >= 50:
            merged.append(curr)
            curr = ""
    if curr:
        if merged:
            merged[-1] += " " + curr
        else:
            merged.append(curr)
            
    if len(merged) > 6:
        # Squash middle sections if too long
        merged = merged[:3] + ["\n".join(merged[3:-1])] + [merged[-1]]
        
    return [m for m in merged if len(m) >= 30]


def _format_as_cot(text: str, label: str, domain: str) -> str:
    """Convert a flat answer into structured CoT format."""
    sections = _split_into_sections(label)
    
    action_sequence = {
        0: "IDENTIFY",
        1: "ANALYZE",
        2: "EVIDENCE",
        3: "EVALUATE",
        -1: "CONCLUDE",
    }
    
    steps = []
    # Ensure safe slicing and cleanup
    q_summary = (text[:100] + "...") if len(text) > 100 else text
    q_summary = q_summary.replace("\n", " ").strip()
    steps.append(f"## Step 1 [IDENTIFY]\nThe query asks about: {q_summary}")
    
    for i, section in enumerate(sections):
        action = action_sequence.get(i + 1, "ANALYZE")
        if i == len(sections) - 1:
            action = "CONCLUDE"
        steps.append(f"## Step {i + 2} [{action}]\n{section.strip()}")
    
    return "\n\n".join(steps)


def _convert_to_cot(records: list, fraction: float = 0.30) -> list:
    """Convert a fraction of records to CoT format deterministically."""
    random.seed(42)  # Deterministic selection
    converted = 0
    total_eligible = 0
    
    for rec in records:
        # Don't convert orchestrator
        if rec.get("domain") == "orchestrator":
            continue
            
        label = rec.get("label", "")
        # Only convert substantial labels with some structure
        if len(label) >= 200 and ("\n" in label or "1." in label):
            total_eligible += 1
            if random.random() < fraction:
                rec["label"] = _format_as_cot(rec.get("text", ""), label, rec.get("domain", ""))
                rec["is_cot"] = True
                converted += 1
                
    print(f"[dataset_loader] Converted {converted}/{total_eligible} eligible records to CoT format ({len(records)} total)")
    return records

# MEDICAL data is now handled externally by scripts/prep_medical_data.py


# ---------------------------------------------------------------------------
# CYBER: MITRE STIX + CyberQA + Trendyol + CyberMetric + Synthetic IR/MITRE
# ---------------------------------------------------------------------------

# Synthetic MITRE ATT&CK Incident Response scenarios — hand-crafted to fix
# the model's inability to correctly map incidents to techniques and follow
# structured IR methodology.
_SYNTHETIC_CYBER_SCENARIOS = [
    {
        "text": "A user reports receiving a phishing email with a malicious Word document attachment. Upon opening it, a macro executed PowerShell commands that downloaded a second-stage payload. Map this to MITRE ATT&CK techniques and recommend containment steps.",
        "label": (
            "## Attack Stage Analysis\n\n"
            "**Initial Access:** The phishing email with a malicious attachment constitutes T1566.001 (Phishing: Spearphishing Attachment).\n\n"
            "**Execution:** The macro in the Word document triggered T1204.002 (User Execution: Malicious File), which then invoked T1059.001 (Command and Scripting Interpreter: PowerShell) to execute commands.\n\n"
            "**Command and Control:** The PowerShell script downloaded a second-stage payload, mapping to T1105 (Ingress Tool Transfer).\n\n"
            "## Containment Steps\n\n"
            "1. **Isolate the endpoint** — Disconnect the affected workstation from the network immediately to prevent lateral movement.\n"
            "2. **Block the C2 domain/IP** — Add the external domain or IP contacted by the PowerShell script to firewall and proxy blocklists.\n"
            "3. **Disable the user's account** — Temporarily disable the compromised user's Active Directory account to prevent credential abuse.\n"
            "4. **Quarantine the email** — Use the email gateway to search for and quarantine all instances of the phishing email across the organization.\n"
            "5. **Capture forensic artifacts** — Preserve PowerShell transcription logs (Event ID 4104), Prefetch files, and a memory image of the workstation before reimaging.\n\n"
            "## Key Distinction\n"
            "Note that T1059.001 (PowerShell) is a *technique* under the Execution *tactic* (TA0002). Tactics describe the adversary's goal (the 'why'), while techniques describe how they achieve it (the 'how'). Do not confuse TA-codes (tactics) with T-codes (techniques)."
        )
    },
    {
        "text": "An EDR alert shows that a scheduled task was created on a domain controller that runs a base64-encoded PowerShell script every 6 hours. The script connects to a Tor exit node. Identify the MITRE ATT&CK techniques and assess the severity.",
        "label": (
            "## MITRE ATT&CK Mapping\n\n"
            "- **T1053.005 (Scheduled Task/Job: Scheduled Task):** The adversary created a scheduled task for persistence, ensuring the payload survives reboots.\n"
            "- **T1059.001 (Command and Scripting Interpreter: PowerShell):** PowerShell is the execution vehicle.\n"
            "- **T1132.001 (Data Encoding: Standard Encoding):** Base64 encoding is used to obfuscate the script content to evade static detection.\n"
            "- **T1090.003 (Proxy: Multi-hop Proxy):** Connecting to a Tor exit node indicates the use of anonymizing proxies for C2 communication.\n"
            "- **T1078.002 (Valid Accounts: Domain Accounts):** Creating a scheduled task on a domain controller likely required domain admin credentials, indicating prior credential compromise.\n\n"
            "## Severity Assessment\n\n"
            "**CRITICAL.** This is a domain controller — the highest-value target in an Active Directory environment. A compromised DC means the adversary likely has:\n"
            "- Access to all domain password hashes (DCSync attack / T1003.006)\n"
            "- Ability to create Golden Tickets (T1558.001) for persistent access\n"
            "- Full control over Group Policy, enabling mass deployment of malware\n\n"
            "The use of Tor for C2 further indicates a sophisticated threat actor attempting to avoid attribution.\n\n"
            "## Immediate Actions\n"
            "1. Isolate the DC from the network (but do NOT power off — preserve volatile memory)\n"
            "2. Capture a full memory dump and disk image for forensics\n"
            "3. Initiate a full domain-wide password reset, starting with KRBTGT (twice)\n"
            "4. Review all scheduled tasks across DCs using `schtasks /query /fo LIST /v`\n"
            "5. Engage incident response team and consider activating the CIRT/CSIRT"
        )
    },
    {
        "text": "A vulnerability scanner identifies CVE-2024-XXXX with a CVSS base score of 9.8 (Critical) on an internal-only application server that is not exposed to the internet and requires VPN + MFA to access. Should this be patched immediately? Provide risk-prioritization reasoning.",
        "label": (
            "## CVSS Base Score vs. Environmental Risk\n\n"
            "The **CVSS base score of 9.8** reflects the worst-case exploitability assuming direct network access, no authentication, and full impact on confidentiality, integrity, and availability. However, CVSS base scores do not account for environmental controls.\n\n"
            "### Environmental Factors That Reduce Real-World Risk:\n"
            "1. **Network accessibility (Modified Attack Vector):** The server is internal-only, not internet-facing. This changes the Attack Vector from 'Network' to effectively 'Adjacent Network' or requires prior internal access — significantly reducing the pool of potential attackers.\n"
            "2. **VPN requirement:** Attackers must first compromise VPN credentials, adding a prerequisite step (T1133 External Remote Services).\n"
            "3. **MFA enforcement:** Even with stolen credentials, the attacker must bypass MFA (T1111 Multi-Factor Authentication Interception), which is a high-effort step.\n\n"
            "### Adjusted Risk Assessment:\n"
            "Using CVSS Environmental scoring, the effective score drops to approximately **6.5–7.5 (High, not Critical)** because:\n"
            "- Modified Attack Vector: Adjacent/Local instead of Network\n"
            "- Modified Privileges Required: High (VPN + MFA = multi-step auth)\n"
            "- Existing compensating controls reduce exploitability\n\n"
            "### Risk-Prioritization Recommendation:\n"
            "**Priority: High, but not Emergency.** Schedule patching within the next maintenance window (7–14 days), not an emergency out-of-band deployment. The reasoning:\n"
            "- The vulnerability IS critical in capability, but the exploitation path requires multiple prerequisite compromises\n"
            "- Rushing an untested patch to production carries its own risk (availability impact)\n"
            "- Monitor IDS/IPS and VPN logs for anomalous access patterns as an interim control\n"
            "- If threat intelligence indicates active exploitation in the wild, escalate to emergency patching"
        )
    },
    {
        "text": "A SIEM alert triggers on a Windows workstation showing: (1) Execution of certutil.exe to download a file, (2) Creation of a new local user, (3) Addition of that user to the Administrators group, (4) RDP connections from the workstation to three other internal hosts. Perform a MITRE ATT&CK analysis.",
        "label": (
            "## MITRE ATT&CK Technique Mapping (Tactic → Technique)\n\n"
            "### Execution (TA0002)\n"
            "- **T1218 (System Binary Proxy Execution):** certutil.exe is a living-off-the-land binary (LOLBin) being abused to download a file, specifically T1105 (Ingress Tool Transfer). Certutil is commonly used because it's a signed Microsoft binary that bypasses application whitelisting.\n\n"
            "### Persistence (TA0003)\n"
            "- **T1136.001 (Create Account: Local Account):** A new local user was created to establish a persistent foothold that survives credential resets of the originally compromised account.\n\n"
            "### Privilege Escalation (TA0004)\n"
            "- **T1098 (Account Manipulation):** Adding the new user to the Administrators group provides elevated privileges without needing to exploit a vulnerability.\n\n"
            "### Lateral Movement (TA0008)\n"
            "- **T1021.001 (Remote Services: Remote Desktop Protocol):** RDP connections to three internal hosts indicate the attacker is moving laterally through the network using the newly escalated privileges.\n\n"
            "### Defense Evasion (TA0005)\n"
            "- **T1218 (System Binary Proxy Execution):** Using certutil.exe instead of PowerShell or curl helps evade detection rules that focus on common download tools.\n\n"
            "## Attack Chain Summary\n"
            "This is a classic post-exploitation sequence: Download tools → Create backdoor account → Escalate privileges → Move laterally. The attacker likely already had initial access (possibly via phishing or an exploit) and is now expanding their foothold.\n\n"
            "## Detection Recommendations\n"
            "1. Alert on certutil.exe with `-urlcache` or `-split` flags (Sigma rule: proc_creation_win_certutil_download)\n"
            "2. Monitor Event ID 4720 (account created) + 4732 (member added to security-enabled local group)\n"
            "3. Baseline normal RDP patterns and alert on new source→destination pairs\n"
            "4. Enable and centralize PowerShell ScriptBlock Logging (Event ID 4104)"
        )
    },
    {
        "text": "Explain the difference between MITRE ATT&CK Tactics and Techniques, and why confusing them is a common mistake in incident reports.",
        "label": (
            "## Tactics vs. Techniques in MITRE ATT&CK\n\n"
            "### Tactics (TA-codes)\n"
            "Tactics represent the **adversary's tactical goal** — the *why* behind an action. They answer: 'What is the attacker trying to achieve at this stage?' There are 14 tactics in Enterprise ATT&CK, including:\n"
            "- TA0001: Initial Access — gaining entry to the network\n"
            "- TA0002: Execution — running malicious code\n"
            "- TA0003: Persistence — maintaining access across reboots\n"
            "- TA0004: Privilege Escalation — gaining higher-level permissions\n"
            "- TA0005: Defense Evasion — avoiding detection\n"
            "- TA0006: Credential Access — stealing credentials\n"
            "- TA0008: Lateral Movement — moving through the network\n"
            "- TA0010: Exfiltration — stealing data\n"
            "- TA0011: Command and Control — communicating with implants\n\n"
            "### Techniques (T-codes)\n"
            "Techniques describe the **specific method** used to achieve a tactic — the *how*. Each tactic contains multiple techniques. For example:\n"
            "- Under TA0002 (Execution): T1059.001 (PowerShell), T1059.003 (Windows Command Shell), T1204 (User Execution)\n"
            "- Under TA0003 (Persistence): T1053.005 (Scheduled Task), T1547.001 (Registry Run Keys)\n\n"
            "### Why Confusion Happens\n"
            "Analysts commonly make these mistakes:\n"
            "1. **Using TA-codes when they mean T-codes:** Writing 'TA0002 PowerShell' when the correct reference is 'T1059.001 PowerShell' under tactic TA0002.\n"
            "2. **Citing techniques without tactic context:** Listing 'T1059.001' without explaining which tactic it supports obscures the attacker's strategic intent.\n"
            "3. **Confusing sub-techniques:** T1059 is the parent technique (Command and Scripting Interpreter), while T1059.001 (PowerShell) is a sub-technique. The sub-technique provides specificity.\n\n"
            "### Best Practice\n"
            "In incident reports, always cite techniques in the format: **T[number].[sub] (Name)** under tactic **TA[number] (Name)**. Example: 'The attacker used T1059.001 (PowerShell) for Execution (TA0002).'"
        )
    },
    {
        "text": "An organization's web application is protected by a Cloudflare WAF and sits behind an Nginx reverse proxy. A penetration tester claims they can exploit an SQL injection vulnerability. Analyze how the WAF and reverse proxy affect the attack surface and what bypasses might be attempted.",
        "label": (
            "## How WAF and Reverse Proxy Affect the Attack Surface\n\n"
            "### Reverse Proxy (Nginx)\n"
            "The reverse proxy provides several security benefits:\n"
            "1. **Origin IP masking:** The application server's real IP is hidden behind Nginx, preventing direct attacks that bypass the WAF.\n"
            "2. **Request filtering:** Nginx can be configured to block unusual HTTP methods, oversized headers, and suspicious URL patterns before they reach the application.\n"
            "3. **SSL/TLS termination:** If SSL terminates at Nginx, the WAF can inspect decrypted traffic. If SSL passes through (end-to-end), the WAF may need to perform SSL interception.\n"
            "4. **Rate limiting:** Nginx rate limiting can slow down automated SQLi enumeration tools like sqlmap.\n\n"
            "### Web Application Firewall (Cloudflare WAF)\n"
            "The WAF provides a managed ruleset that detects common attack patterns:\n"
            "1. **Signature-based detection:** Blocks known SQLi patterns like `' OR 1=1--`, `UNION SELECT`, and `SLEEP()` commands.\n"
            "2. **Anomaly scoring:** Assigns risk scores to requests based on multiple indicators, blocking requests that exceed a threshold.\n"
            "3. **Virtual patching:** Can deploy rules for specific CVEs before the application is patched.\n\n"
            "### Potential WAF Bypass Techniques (T1190 — Exploit Public-Facing Application)\n"
            "A skilled attacker may attempt:\n"
            "1. **Encoding evasion:** Using double URL encoding, Unicode encoding, or hex encoding to obfuscate SQLi payloads (e.g., `%2527` instead of `'`).\n"
            "2. **Chunked transfer encoding:** Splitting the malicious payload across multiple HTTP chunks to evade pattern matching.\n"
            "3. **HTTP Parameter Pollution (HPP):** Sending duplicate parameters where the WAF checks one value but the application processes another.\n"
            "4. **JSON/XML body injection:** If the WAF primarily inspects URL parameters but the application accepts JSON bodies, injecting SQL through JSON fields.\n"
            "5. **Time-based blind SQLi:** Using `WAITFOR DELAY` or `BENCHMARK()` which may not trigger signature rules as aggressively as `UNION SELECT`.\n"
            "6. **Direct-to-origin:** If the attacker discovers the origin server's IP (via DNS history, SSL certificates, or information leakage), they can bypass the WAF entirely.\n\n"
            "### Defense Recommendations\n"
            "1. Ensure the origin server only accepts connections from the WAF/proxy IP ranges\n"
            "2. Enable WAF logging in 'detection' mode first to tune rules before blocking\n"
            "3. Implement parameterized queries / prepared statements at the application level — WAFs are defense-in-depth, not a substitute for secure coding\n"
            "4. Regularly test WAF rules with tools like WAFNinja or bypass wordlists"
        )
    },
    {
        "text": "A Windows server begins communicating with an unknown external IP address every 15 minutes. Investigation reveals that a PowerShell script was executed from a user's Downloads folder. Shortly afterward, a new local administrator account was created and several compressed archives containing sensitive files were transferred to the external IP. Perform a complete incident response analysis following NIST SP 800-61 phases.",
        "label": (
            "## Incident Response Analysis — NIST SP 800-61 Framework\n\n"
            "### Phase 1: Identification\n"
            "The following indicators of compromise (IOCs) have been identified:\n"
            "- **Beaconing behavior:** Regular 15-minute interval communications to an unknown external IP — consistent with C2 beacon timing (T1071.001 Application Layer Protocol: Web Protocols, T1573 Encrypted Channel).\n"
            "- **Unauthorized script execution:** PowerShell script from Downloads folder — T1059.001 (Command and Scripting Interpreter: PowerShell). Execution from Downloads strongly suggests user-initiated execution of a socially engineered payload (T1204.002 User Execution: Malicious File).\n"
            "- **Privilege escalation:** Creation of a new local administrator account — T1136.001 (Create Account: Local Account) + T1098 (Account Manipulation).\n"
            "- **Data staging and exfiltration:** Compressed archives of sensitive files transferred externally — T1560.001 (Archive Collected Data: Archive via Utility) + T1041 (Exfiltration Over C2 Channel).\n\n"
            "### Phase 2: Containment\n"
            "**Short-term containment (immediate — within minutes):**\n"
            "1. Isolate the server from the network by disabling the NIC or moving to an isolated VLAN — do NOT power off to preserve memory forensics.\n"
            "2. Block the external IP at the perimeter firewall (both ingress and egress).\n"
            "3. Disable the unauthorized administrator account immediately.\n"
            "4. Revoke active sessions for the compromised user account.\n\n"
            "**Long-term containment (within hours):**\n"
            "1. Deploy EDR to monitor for any additional compromised endpoints.\n"
            "2. Implement network monitoring for any other hosts communicating with the same IP.\n"
            "3. Rotate credentials for all accounts that had access to the compromised server.\n\n"
            "### Phase 3: Eradication\n"
            "1. Remove the malicious PowerShell script and any dropped payloads.\n"
            "2. Delete the unauthorized administrator account and audit all recent account creations (Event ID 4720).\n"
            "3. Remove any persistence mechanisms: check scheduled tasks (T1053.005), registry Run keys (T1547.001), startup folders, and WMI subscriptions (T1546.003).\n"
            "4. Scan all endpoints with updated IOCs (file hashes, IP addresses, domain names).\n\n"
            "### Phase 4: Recovery\n"
            "1. Reimage the compromised server from a known-good baseline.\n"
            "2. Restore sensitive files from verified clean backups.\n"
            "3. Gradually reconnect the server to the network with enhanced monitoring.\n"
            "4. Validate that no beaconing resumes after reconnection.\n\n"
            "### Phase 5: Lessons Learned\n"
            "1. Conduct a post-incident review within 72 hours.\n"
            "2. Assess why the PowerShell script was able to execute from Downloads — implement AppLocker or WDAC policies.\n"
            "3. Evaluate whether DLP controls could have detected the archive exfiltration.\n"
            "4. Update detection rules in SIEM/EDR for the identified TTPs.\n"
            "5. Document timeline and share sanitized IOCs with ISACs.\n\n"
            "### MITRE ATT&CK Summary\n"
            "| Tactic | Technique ID | Technique Name |\n"
            "| --- | --- | --- |\n"
            "| Initial Access (TA0001) | T1204.002 | User Execution: Malicious File |\n"
            "| Execution (TA0002) | T1059.001 | PowerShell |\n"
            "| Persistence (TA0003) | T1136.001 | Create Account: Local Account |\n"
            "| Privilege Escalation (TA0004) | T1098 | Account Manipulation |\n"
            "| Collection (TA0009) | T1560.001 | Archive Collected Data |\n"
            "| Command and Control (TA0011) | T1071.001 | Application Layer Protocol |\n"
            "| Exfiltration (TA0010) | T1041 | Exfiltration Over C2 Channel |"
        )
    },
    {
        "text": "During a forensic investigation of a ransomware attack, you find the following artifacts: (1) A Windows Event Log entry showing PsExec execution, (2) Deleted shadow copies via vssadmin, (3) A ransom note in every directory, (4) Encrypted files with .locked extension. Map each artifact to MITRE ATT&CK techniques.",
        "label": (
            "## Artifact-to-MITRE ATT&CK Mapping\n\n"
            "### Artifact 1: PsExec Execution (Event Log)\n"
            "- **T1569.002 (System Services: Service Execution):** PsExec works by creating a temporary Windows service on the remote host to execute commands. Look for Event ID 7045 (new service installed) in the System log.\n"
            "- **T1021.002 (Remote Services: SMB/Windows Admin Shares):** PsExec uses SMB (port 445) and the ADMIN$ share to copy its service binary to the target. This maps to Lateral Movement (TA0008).\n"
            "- **T1570 (Lateral Tool Transfer):** PsExec transfers its executable to the remote system via the admin share.\n\n"
            "### Artifact 2: Deleted Shadow Copies (vssadmin)\n"
            "- **T1490 (Inhibit System Recovery):** Deleting shadow copies with `vssadmin delete shadows /all /quiet` prevents victims from restoring files from Volume Shadow Copies, which is a standard ransomware pre-encryption step.\n"
            "- **T1059.003 (Command and Scripting Interpreter: Windows Command Shell):** vssadmin is typically invoked via cmd.exe or PowerShell.\n\n"
            "### Artifact 3: Ransom Note in Every Directory\n"
            "- **T1486 (Data Encrypted for Impact):** The ransom note is the direct indicator of the ransomware's impact phase, placed alongside encrypted files to instruct victims on payment.\n"
            "- This maps to the Impact tactic (TA0040).\n\n"
            "### Artifact 4: Encrypted Files (.locked extension)\n"
            "- **T1486 (Data Encrypted for Impact):** The file encryption itself, with the .locked extension serving as a visual indicator to the victim.\n"
            "- **T1083 (File and Directory Discovery):** Before encryption, the ransomware enumerated files and directories to target valuable data while skipping system-critical files.\n\n"
            "## Kill Chain Reconstruction\n"
            "1. **Lateral Movement:** Attacker used PsExec to move to additional hosts (T1021.002)\n"
            "2. **Defense Evasion/Impact Prep:** Deleted shadow copies to prevent recovery (T1490)\n"
            "3. **Impact:** Encrypted files and dropped ransom notes (T1486)\n\n"
            "## Key Forensic Evidence to Preserve\n"
            "- Windows Event Logs: System (Event ID 7045), Security (Event ID 4624 Type 3 for network logon)\n"
            "- Prefetch files for PsExec and vssadmin\n"
            "- Master File Table ($MFT) for timestamp analysis of encrypted files\n"
            "- Network logs showing SMB lateral movement patterns"
        )
    },
    {
        "text": "Compare and contrast a vulnerability assessment and a penetration test. When should an organization use each, and how do they complement each other in a security program?",
        "label": (
            "## Vulnerability Assessment vs. Penetration Test\n\n"
            "### Vulnerability Assessment\n"
            "**Purpose:** Identify and catalog known vulnerabilities across an environment.\n"
            "**Methodology:** Automated scanning tools (Nessus, Qualys, OpenVAS) compare system configurations, software versions, and exposed services against vulnerability databases (CVE/NVD).\n"
            "**Output:** A prioritized list of vulnerabilities with CVSS scores, affected assets, and remediation guidance.\n"
            "**Scope:** Broad — aims to cover the entire attack surface.\n"
            "**Risk level:** Low — non-exploitative, does not attempt to breach systems.\n"
            "**Frequency:** Regular (monthly/quarterly) as part of continuous monitoring.\n\n"
            "### Penetration Test\n"
            "**Purpose:** Simulate a real-world attack to determine whether vulnerabilities can actually be exploited and what damage could result.\n"
            "**Methodology:** Manual testing by skilled ethical hackers who chain vulnerabilities together, attempt privilege escalation, lateral movement, and data exfiltration — following methodologies like PTES or OWASP Testing Guide.\n"
            "**Output:** A narrative report showing the attack path, business impact, and evidence of compromise (screenshots, data accessed).\n"
            "**Scope:** Targeted — focuses on specific systems, applications, or scenarios.\n"
            "**Risk level:** Moderate — actively exploits vulnerabilities, may cause service disruption.\n"
            "**Frequency:** Annual or after significant changes (new application, infrastructure migration).\n\n"
            "### When to Use Each\n"
            "- **Vulnerability Assessment:** Use for routine hygiene, compliance requirements (PCI DSS requires quarterly scans), and to maintain visibility of your attack surface.\n"
            "- **Penetration Test:** Use to validate security controls, test incident response processes, and demonstrate real-world risk to stakeholders. Required annually by PCI DSS, SOC 2, and many regulatory frameworks.\n\n"
            "### How They Complement Each Other\n"
            "1. **VA feeds Pentest scope:** Vulnerability assessments identify the most critical findings, which penetration testers can then attempt to exploit in a focused engagement.\n"
            "2. **Pentest validates VA priority:** A CVSS 9.8 vulnerability may be unexploitable due to environmental controls — a pentest confirms (or denies) this.\n"
            "3. **Continuous improvement loop:** VA provides breadth (scanning everything regularly), Pentest provides depth (proving what's actually exploitable).\n"
            "4. **Together they satisfy compliance:** Most frameworks require both: regular scanning AND periodic penetration testing."
        )
    },
    {
        "text": "What forensic artifacts should be collected from a compromised Windows endpoint, in what order, and why does the order of volatility matter?",
        "label": (
            "## Order of Volatility in Digital Forensics\n\n"
            "The **order of volatility** (RFC 3227) dictates that forensic evidence should be collected starting with the most volatile (easily lost) artifacts first. This is critical because:\n"
            "- Volatile data is lost when the system is powered off or rebooted\n"
            "- Attackers may have persistence mechanisms that activate on reboot, destroying evidence\n"
            "- Memory-resident malware (fileless malware, T1055 Process Injection) exists only in RAM\n\n"
            "## Collection Order (Most → Least Volatile)\n\n"
            "### 1. CPU Registers & Cache (seconds)\n"
            "- Captured automatically during memory acquisition\n"
            "- Contains currently executing instructions\n\n"
            "### 2. RAM / Physical Memory (seconds to minutes)\n"
            "- **Tool:** WinPMEM, FTK Imager, DumpIt\n"
            "- **What it reveals:** Running processes, network connections, decrypted data, injected code, clipboard contents, command history\n"
            "- **Why critical:** Fileless malware (T1059.001 PowerShell, T1055 Process Injection) may exist ONLY in memory\n\n"
            "### 3. Network State (seconds to minutes)\n"
            "- **Commands:** `netstat -anob`, `ipconfig /all`, `arp -a`, `dns cache`\n"
            "- **What it reveals:** Active C2 connections, listening backdoor ports, DNS cache showing resolved malicious domains\n\n"
            "### 4. Running Processes (minutes)\n"
            "- **Commands:** `tasklist /v`, `wmic process list full`\n"
            "- **Tools:** Process Monitor, Sysmon logs\n"
            "- **What it reveals:** Malicious processes, parent-child process trees, loaded DLLs\n\n"
            "### 5. Disk — Filesystem Metadata (hours)\n"
            "- **Artifacts:** $MFT (Master File Table), $UsnJrnl (Change Journal), Prefetch files, Amcache, ShimCache\n"
            "- **What it reveals:** File creation/modification timestamps, executed programs, deleted file traces\n\n"
            "### 6. Disk — Full Image (stable)\n"
            "- **Tool:** FTK Imager, dd\n"
            "- **Create a bit-for-bit forensic image** with hash verification (MD5 + SHA256)\n"
            "- **What it reveals:** All files including deleted ones (recoverable from unallocated space), browser history, registry hives\n\n"
            "### 7. Windows Event Logs (stable)\n"
            "- **Key logs:** Security.evtx, System.evtx, PowerShell/Operational.evtx, Sysmon/Operational.evtx\n"
            "- **Key Event IDs:** 4624 (logon), 4720 (account created), 4732 (group membership changed), 4104 (PowerShell ScriptBlock), 7045 (new service)\n\n"
            "### 8. External Logs (stable)\n"
            "- Firewall logs, proxy logs, DNS query logs, DHCP logs, Active Directory replication logs\n"
            "- These exist on external systems and are the least volatile\n\n"
            "## Key Principle\n"
            "Always collect MORE volatile evidence FIRST. A common mistake is reimaging the machine immediately, which destroys all volatile artifacts (RAM, network state, running processes) before they can be analyzed."
        )
    },
    {
        "text": "An organization detects a successful brute force attack against a public-facing SSH server. The attacker successfully logged in as 'root' and established an interactive shell. The organization wants to contain the incident. Draft a detailed containment and eradication playbook.",
        "label": (
            "## Incident Response Playbook: Compromised Public-Facing SSH Server\n\n"
            "### Phase 2: Containment\n"
            "**Short-term (Immediate Containment):**\n"
            "1. **Isolate the Server:** Immediately drop all inbound and outbound traffic to the server via edge firewall rules, except for forensic investigator IPs.\n"
            "2. **Session Termination:** Log into the console (out-of-band management like iLO/iDRAC) and forcefully kill the attacker's shell session (`kill -9 <PID>` of the bash session spawned by SSHd).\n"
            "3. **Block IP:** Add the attacker's source IP address to the border firewall drop list.\n"
            "4. **Preserve State:** Do NOT reboot the server, as this will clear RAM where the attacker may have injected fileless malware or memory-resident rootkits.\n\n"
            "**Long-term (Sustained Containment):**\n"
            "1. **Disable Root SSH Login:** Modify `/etc/ssh/sshd_config` to set `PermitRootLogin no` globally across all servers.\n"
            "2. **Enforce Key-Based Auth:** Disable password authentication (`PasswordAuthentication no`) to prevent future brute force attacks.\n"
            "3. **Audit Lateral Movement:** Review `~/.ssh/known_hosts` and `~/.ssh/authorized_keys` to see if the attacker pivoted to or established persistence on other internal hosts.\n\n"
            "### Phase 3: Eradication\n"
            "1. **Identify Dropped Payloads:** Search for recently created files (`find / -mmin -60`) and unauthorized SUID binaries.\n"
            "2. **Remove Persistence:** Check crontabs (`crontab -l` for all users) and systemd services for malicious persistence mechanisms.\n"
            "3. **Wipe and Rebuild:** Because the attacker achieved root access, the server cannot be trusted. After capturing forensic images, the server must be completely wiped and rebuilt from a known-good immutable image.\n\n"
            "### Phase 4: Recovery\n"
            "1. Re-deploy the server from IaC (Infrastructure as Code) templates.\n"
            "2. Restore application data from backups taken *prior* to the intrusion.\n"
            "3. Place the SSH interface behind a VPN or Zero Trust Network Access (ZTNA) proxy rather than exposing it directly to the internet."
        )
    }
]

def fetch_cyber():
    print("[dataset_loader] === CYBER DATASET (v2 — Hardened) ===")
    records = []

    # 1. MITRE ATT&CK STIX — attack patterns, malware, intrusion sets
    try:
        print("[dataset_loader] [1/5] Downloading MITRE ATT&CK STIX data...")
        url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
        response = requests.get(url)
        response.raise_for_status()
        stix_data = response.json()

        objects = stix_data.get("objects", [])
        added = 0
        for obj in objects:
            if obj.get("type") == "attack-pattern":
                name = obj.get("name", "")
                description = obj.get("description", "")
                if name and description and len(description) >= 30:
                    records.append({
                        "id": f"cyb_{uuid.uuid4().hex[:8]}",
                        "text": f"Explain the cyber threat tactic/technique: {name}.",
                        "label": description,
                        "domain": "cyber"
                    })
                    added += 1
            elif obj.get("type") in ["malware", "intrusion-set"]:
                name = obj.get("name", "")
                description = obj.get("description", "")
                if name and description and len(description) >= 30:
                    records.append({
                        "id": f"cyb_{uuid.uuid4().hex[:8]}",
                        "text": f"Provide threat intelligence on: {name}.",
                        "label": description,
                        "domain": "cyber"
                    })
                    added += 1
        print(f"[dataset_loader]   STIX: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading STIX data: {e}")

    # 2. infosec-security-qa — conversational Q&A
    try:
        print("[dataset_loader] [2/5] Downloading infosec-security-qa dataset...")
        ds = load_dataset("pAILabs/infosec-security-qa", split="train[:5000]")
        added = 0
        for item in ds:
            question = item.get("question", "")
            answer = item.get("answer", "")
            if question and answer and len(answer) >= 30:
                records.append({
                    "id": f"cyb_{uuid.uuid4().hex[:8]}",
                    "text": question,
                    "label": answer,
                    "domain": "cyber"
                })
                added += 1
        print(f"[dataset_loader]   infosec-security-qa: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading infosec-security-qa: {e}")

    # 3. Threat Intelligence instruction dataset (supplement)
    try:
        print("[dataset_loader] [3/5] Downloading Threat Intelligence dataset...")
        ds3 = load_dataset("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", split="train[:5000]")
        added = 0
        for item in ds3:
            instruction = item.get("user", "") or item.get("instruction", "")
            output = item.get("assistant", "") or item.get("output", "")
            if instruction and output and len(output) >= 30:
                records.append({
                    "id": f"cyb_{uuid.uuid4().hex[:8]}",
                    "text": instruction,
                    "label": output,
                    "domain": "cyber"
                })
                added += 1
        print(f"[dataset_loader]   Trendyol: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading Threat Intelligence data: {e}")

    # 4. CyberMetric — expert-level cybersecurity benchmark (NIST, ISO, IR, CVSS)
    try:
        print("[dataset_loader] [4/5] Downloading CyberMetric benchmark...")
        # Try different configs — this dataset may have subsets
        ds4 = None
        for config_name in [None, "CyberMetric-2000", "CyberMetric-500", "default"]:
            try:
                if config_name:
                    ds4 = load_dataset("AcerSeb/CyberMetric", config_name, split="train[:3000]")
                else:
                    ds4 = load_dataset("AcerSeb/CyberMetric", split="train[:3000]")
                print(f"[dataset_loader]   CyberMetric loaded with config={config_name}, columns={ds4.column_names}")
                break
            except Exception:
                continue

        added = 0
        if ds4 is not None:
            for item in ds4:
                # Probe for field names
                question = (item.get("question", "") or item.get("Question", "") or
                           item.get("input", "") or item.get("prompt", ""))
                answer = (item.get("answer", "") or item.get("Answer", "") or
                         item.get("output", "") or item.get("response", ""))
                explanation = (item.get("explanation", "") or item.get("Explanation", "") or
                              item.get("rationale", ""))

                # Try to extract from choices/options if MCQ
                choices = item.get("choices", []) or item.get("options", [])
                correct_idx = item.get("correct_answer", None) or item.get("answer_idx", None)

                if question and answer:
                    label = answer
                    if explanation:
                        label = f"{answer}\n\nExplanation: {explanation}"
                    if len(label) >= 30:
                        records.append({
                            "id": f"cyb_{uuid.uuid4().hex[:8]}",
                            "text": question,
                            "label": label,
                            "domain": "cyber"
                        })
                        added += 1
        print(f"[dataset_loader]   CyberMetric: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading CyberMetric: {e}")

    # 5. Synthetic MITRE ATT&CK / IR scenarios (hand-crafted, high quality)
    print("[dataset_loader] [5/5] Adding synthetic MITRE/IR scenarios...")
    for scenario in _SYNTHETIC_CYBER_SCENARIOS:
        records.append({
            "id": f"cyb_syn_{uuid.uuid4().hex[:8]}",
            "text": scenario["text"],
            "label": scenario["label"],
            "domain": "cyber"
        })
    print(f"[dataset_loader]   Synthetic: {len(_SYNTHETIC_CYBER_SCENARIOS)} records")

    records = _quality_filter(records)
    records = _convert_to_cot(records, fraction=0.30)
    _jsonl_write(records, "data/processed/cyber.jsonl")


# ---------------------------------------------------------------------------
# SCIENCE: ScienceQA + SciQ + Hendrycks MATH + ARC-Challenge + CAMEL-AI (physics/chemistry/biology)
# ---------------------------------------------------------------------------

def _extract_camel_ai(dataset_name: str, subject: str, max_records: int = 2000) -> list:
    """Extract Q&A pairs from CAMEL-AI dialogue datasets."""
    records = []
    try:
        ds = load_dataset(dataset_name, split=f"train[:{max_records}]")
        for item in ds:
            q = (item.get("message_1", "") or item.get("instruction", "") or
                 item.get("input", ""))
            a = (item.get("message_2", "") or item.get("output", "") or
                 item.get("response", ""))
            if isinstance(q, dict):
                q = q.get("content", "")
            if isinstance(a, dict):
                a = a.get("content", "")

            # Strict negative filter for AI hedging
            bad_phrases = ["i might be wrong", "wait, let me", "correction:", "as an ai", "let me re-evaluate", "sorry,"]
            
            if q and a and len(a) >= 50:
                if not any(bp in a.lower() for bp in bad_phrases):
                    records.append({
                        "id": f"sci_{uuid.uuid4().hex[:8]}",
                        "text": f"[{subject}] {q}" if not q.startswith("[") else q,
                        "label": a,
                        "domain": "science"
                    })
    except Exception as e:
        print(f"[dataset_loader] Error downloading {dataset_name}: {e}")
    return records


def fetch_science():
    print("[dataset_loader] === SCIENCE DATASET (v2 — Expanded) ===")
    records = []

    # 2. ScienceQA — multi-subject science with explanations
    try:
        print("[dataset_loader] [2/8] Downloading ScienceQA...")
        ds_sci = load_dataset("lmms-lab/ScienceQA", "ScienceQA-FULL", split="validation[:5000]")
        added = 0
        for item in ds_sci:
            question = item.get("question", "")
            choices = item.get("choices", [])
            answer_idx = item.get("answer", -1)
            lecture = item.get("lecture", "")
            solution = item.get("solution", "")
            if not question or not choices or answer_idx < 0 or answer_idx >= len(choices):
                continue
            if not solution or len(solution) < 20:
                continue
            choice_letters = ["A", "B", "C", "D", "E", "F"]
            choices_text = "\n".join(
                f"{choice_letters[i]}) {c}" for i, c in enumerate(choices) if i < len(choice_letters)
            )
            full_question = f"{question}\n{choices_text}"
            correct_letter = choice_letters[answer_idx] if answer_idx < len(choice_letters) else "?"
            correct_text = choices[answer_idx]
            full_answer = f"The correct answer is {correct_letter}) {correct_text}.\n\n"
            if lecture:
                full_answer += f"Background: {lecture}\n\n"
            full_answer += f"Solution: {solution}"
            records.append({
                "id": f"sci_{uuid.uuid4().hex[:8]}",
                "text": full_question,
                "label": full_answer,
                "domain": "science"
            })
            added += 1
        print(f"[dataset_loader]   ScienceQA: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading ScienceQA: {e}")

    # 3. SciQ — Science exam questions with support paragraphs
    try:
        print("[dataset_loader] [3/8] Downloading SciQ (allenai)...")
        ds_sciq = load_dataset("allenai/sciq", split="train[:5000]")
        added = 0
        for item in ds_sciq:
            question = item.get("question", "")
            correct = item.get("correct_answer", "")
            support = item.get("support", "")
            d1 = item.get("distractor1", "")
            d2 = item.get("distractor2", "")
            d3 = item.get("distractor3", "")
            if not question or not correct:
                continue
            if d1 and d2 and d3:
                import random
                options = [correct, d1, d2, d3]
                random.shuffle(options)
                correct_idx = options.index(correct)
                choice_letters = ["A", "B", "C", "D"]
                choices_text = "\n".join(f"{choice_letters[i]}) {opt}" for i, opt in enumerate(options))
                full_question = f"[Science] {question}\n{choices_text}"
                full_answer = f"The correct answer is {choice_letters[correct_idx]}) {correct}."
            else:
                full_question = f"[Science] {question}"
                full_answer = correct
            if support and len(support) > 20:
                full_answer += f"\n\nExplanation: {support}"
            if len(full_answer) >= 30:
                records.append({
                    "id": f"sci_{uuid.uuid4().hex[:8]}",
                    "text": full_question,
                    "label": full_answer,
                    "domain": "science"
                })
                added += 1
        print(f"[dataset_loader]   SciQ: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading SciQ: {e}")

    # 4. Hendrycks MATH (STEM Reasoning)
    try:
        print("[dataset_loader] [4/8] Downloading Hendrycks MATH...")
        ds_math_h = load_dataset("competition_math", split="train[:5000]")
        added = 0
        for item in ds_math_h:
            question = item.get("problem", "")
            solution = item.get("solution", "")
            if question and solution:
                records.append({
                    "id": f"sci_{uuid.uuid4().hex[:8]}",
                    "text": f"[Advanced Math] {question}",
                    "label": solution,
                    "domain": "science"
                })
                added += 1
        print(f"[dataset_loader]   MATH: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading Hendrycks MATH: {e}")

    # 5. ARC-Challenge
    try:
        print("[dataset_loader] [5/8] Downloading ARC-Challenge...")
        ds_arc = load_dataset("ai2_arc", "ARC-Challenge", split="train[:3000]")
        added = 0
        for item in ds_arc:
            question = item.get("question", "")
            choices = item.get("choices", {})
            answer_key = item.get("answerKey", "")
            if question and choices and answer_key:
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                choices_text = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
                full_question = f"[Science] {question}\n{choices_text}"
                correct_idx = labels.index(answer_key) if answer_key in labels else -1
                if correct_idx >= 0:
                    full_answer = f"The correct answer is {answer_key}) {texts[correct_idx]}."
                    records.append({
                        "id": f"sci_{uuid.uuid4().hex[:8]}",
                        "text": full_question,
                        "label": full_answer,
                        "domain": "science"
                    })
                    added += 1
        print(f"[dataset_loader]   ARC-Challenge: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading ARC-Challenge: {e}")

    # 6. TIGER-Lab MathInstruct (Physics filtered)
    try:
        print("[dataset_loader] [6/8] Downloading MathInstruct (Physics subset)...")
        ds_math = load_dataset("TIGER-Lab/MathInstruct", split="train[:15000]")
        added = 0
        for item in ds_math:
            question = item.get("instruction", "")
            ans = item.get("output", "")
            if question and ans and any(kw in question.lower() for kw in ["velocity", "friction", "gravity", "acceleration", "mass", "joule", "newton"]):
                records.append({
                    "id": f"sci_{uuid.uuid4().hex[:8]}",
                    "text": f"Solve the following mathematical physics problem step-by-step:\n{question}",
                    "label": ans,
                    "domain": "science"
                })
                added += 1
        print(f"[dataset_loader]   MathInstruct (Physics): {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading MathInstruct: {e}")

    # 7. CAMEL-AI Sciences
    print("[dataset_loader] [7/8] Downloading CAMEL-AI Physics...")
    physics_recs = _extract_camel_ai("camel-ai/physics", "Physics", 8000)
    records.extend(physics_recs)
    print(f"[dataset_loader]   CAMEL-AI Physics: {len(physics_recs)} records")

    print("[dataset_loader] [8/8] Downloading CAMEL-AI Chemistry/Biology...")
    records.extend(_extract_camel_ai("camel-ai/chemistry", "Chemistry", 5000))
    records.extend(_extract_camel_ai("camel-ai/biology", "Biology", 5000))

    records = _quality_filter(records, min_label_len=50)
    records = _convert_to_cot(records, fraction=0.30)
    _jsonl_write(records, "data/processed/science.jsonl")


# ---------------------------------------------------------------------------
# CODING: 60% Algorithmic / 40% Syntax
# ---------------------------------------------------------------------------

def fetch_coding():
    print("[dataset_loader] === CODING DATASET ===")
    records = []

    # 1. Python Syntax / Implementation (Capped at 10K total)
    try:
        print("[dataset_loader] [1/5] Downloading Python Syntax Datasets (Capped)...")
        ds1 = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:5000]")
        added = 0
        for item in ds1:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            text = f"{instruction}\n{input_text}".strip()
            if text and output and len(output) >= 30:
                records.append({
                    "id": f"code_{uuid.uuid4().hex[:8]}",
                    "text": text,
                    "label": output,
                    "domain": "coding"
                })
                added += 1
                
        ds2 = load_dataset("flytech/python-codes-25k", split="train[:5000]")
        for item in ds2:
            text = item.get("text", "")
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            if not instruction and "```python" in text:
                parts = text.split("```python", 1)
                instruction = parts[0].strip()
                output = "```python" + parts[1]
            if instruction and output and len(output) >= 30:
                records.append({
                    "id": f"code_{uuid.uuid4().hex[:8]}",
                    "text": instruction,
                    "label": output,
                    "domain": "coding"
                })
                added += 1
        print(f"[dataset_loader]   Python Syntax: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading Python Syntax: {e}")

    # 2. APPS (Competition Level Algorithmic)
    try:
        print("[dataset_loader] [2/5] Downloading APPS (Competition Level)...")
        ds_apps = load_dataset("codeparrot/apps", split="train[:20000]", trust_remote_code=True)
        added = 0
        for item in ds_apps:
            if item.get("difficulty") == "competition":
                question = item.get("question", "")
                solutions = item.get("solutions", "[]")
                try:
                    sols = json.loads(solutions)
                    if sols and len(sols) > 0:
                        records.append({
                            "id": f"code_{uuid.uuid4().hex[:8]}",
                            "text": f"Solve this competitive programming problem:\n{question}",
                            "label": f"```python\n{sols[0]}\n```",
                            "domain": "coding"
                        })
                        added += 1
                except:
                    pass
                if added >= 8000:
                    break
        print(f"[dataset_loader]   APPS Competition: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading APPS: {e}")

    # 3. CodeContests (DeepMind)
    try:
        print("[dataset_loader] [3/5] Downloading DeepMind CodeContests...")
        ds_cc = load_dataset("deepmind/code_contests", split="train[:5000]")
        added = 0
        for item in ds_cc:
            desc = item.get("description", "")
            sols = item.get("solutions", {})
            python_sols = sols.get("language", [])
            idx = -1
            for i, lang in enumerate(python_sols):
                if lang == 3: # Python 3
                    idx = i
                    break
            if desc and idx >= 0:
                solution_code = sols.get("solution", [])[idx]
                records.append({
                    "id": f"code_{uuid.uuid4().hex[:8]}",
                    "text": f"Solve this algorithmic puzzle:\n{desc}",
                    "label": f"```python\n{solution_code}\n```",
                    "domain": "coding"
                })
                added += 1
        print(f"[dataset_loader]   CodeContests: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading CodeContests: {e}")

    # 4. LeetCode Hard
    try:
        print("[dataset_loader] [4/5] Downloading LeetCode Hard...")
        ds_lc = load_dataset("greengerong/leetcode", split="train")
        added = 0
        for item in ds_lc:
            if item.get("difficulty") == "Hard":
                q = item.get("content", "")
                ans = item.get("python", "")
                if q and ans:
                    records.append({
                        "id": f"code_{uuid.uuid4().hex[:8]}",
                        "text": f"Solve optimally:\n{q}",
                        "label": f"```python\n{ans}\n```",
                        "domain": "coding"
                    })
                    added += 1
        print(f"[dataset_loader]   LeetCode Hard: {added} records (Dataset limited)")
    except Exception as e:
        print(f"[dataset_loader] Error downloading LeetCode: {e}")

    # 5. CodeFeedback (Coding Split)
    try:
        print("[dataset_loader] [5/5] Downloading CodeFeedback (Coding Split)...")
        ds_cf = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train[:15000]")
        added = 0
        coding_keywords = ["debug", "algorithm", "time complexity", "memory", "optimize", "review", "sort", "search"]
        arch_keywords = ["system design", "infrastructure", "scalable", "microservice", "load balancer", "kubernetes", "aws", "docker"]
        
        for item in ds_cf:
            q = item.get("query", "")
            a = item.get("answer", "")
            q_lower = q.lower()
            
            is_coding = any(k in q_lower for k in coding_keywords)
            is_arch = any(k in q_lower for k in arch_keywords)
            
            # Strict router: MUST be coding, MUST NOT be architecture
            if is_coding and not is_arch and len(a) >= 100:
                records.append({
                    "id": f"code_cf_{uuid.uuid4().hex[:8]}",
                    "text": q,
                    "label": a,
                    "domain": "coding"
                })
                added += 1
        print(f"[dataset_loader]   CodeFeedback Coding: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading CodeFeedback: {e}")

    records = _quality_filter(records)
    records = _convert_to_cot(records, fraction=0.30)
    _jsonl_write(records, "data/processed/coding.jsonl")


# ---------------------------------------------------------------------------
# ARCHITECTURE: System Design & Secure Architecture
# ---------------------------------------------------------------------------

def generate_synthetic_architecture():
    """Procedurally generates diverse architecture scenarios based on 8 templates."""
    import random
    scenarios = []
    
    cloud_providers = ["AWS", "GCP", "Azure", "On-Premises", "Hybrid Cloud"]
    scale_levels = ["10,000", "1 million", "50 million", "global scale"]
    industries = ["e-commerce", "fintech", "healthcare", "streaming media", "gaming", "SaaS"]
    compliance_standards = ["HIPAA", "SOC2", "PCI-DSS", "GDPR", "FedRAMP"]
    team_sizes = ["startup (5 engineers)", "mid-size (20 engineers)", "enterprise (100+ engineers)"]
    legacy_systems = ["Oracle DB", "on-prem VMware", "mainframe COBOL", "monolithic Java"]
    
    templates = [
        # Template 1: Cloud migration
        lambda c, s, i, comp, t, l: (
            f"Plan a cloud migration for a {i} company moving from {l} to {c}. Team size is {t} and they need to maintain {comp} compliance.",
            f"## Cloud Migration Strategy for {i.title()} to {c}\n\n**1. Assessment & Discovery:**\nEvaluate the existing {l} footprint. Identify dependencies and map out data gravity. Ensure {comp} requirements are documented as migration constraints.\n\n**2. Migration Strategy (Strangler Fig):**\nUse the Strangler pattern to incrementally migrate services from {l} to {c}. Since the team is {t}, start with low-risk stateless services to build cloud-native muscles.\n\n**3. Phased Execution:**\nPhase 1: Rehost (Lift and Shift) non-critical apps. Phase 2: Replatform databases to managed services. Phase 3: Refactor core business logic into microservices.\n\n**4. Rollback & Validation:**\nRun parallel environments. Route 5% of traffic to the new {c} stack and validate functional parity and {comp} compliance before full cutover."
        ),
        # Template 2: Microservices decomposition
        lambda c, s, i, comp, t, l: (
            f"How should we decompose our {l} monolith for an {i} platform targeting {s} users on {c}?",
            f"## Microservices Decomposition Strategy\n\n**1. Monolith Analysis:**\nThe {l} system has tightly coupled domains. Use Domain-Driven Design (DDD) to identify bounded contexts within the {i} domain.\n\n**2. Service Boundaries:**\nExtract services by business capability (e.g., User Management, Billing, Inventory). For {s} users, ensure each service can scale independently on {c}.\n\n**3. Communication Patterns:**\nUse asynchronous messaging (e.g., Kafka or RabbitMQ) for inter-service communication to reduce temporal coupling and improve fault tolerance. Synchronous REST/gRPC only for direct client-facing APIs.\n\n**4. Data Ownership:**\nDatabase-per-service pattern. Extract data from the legacy {l} datastore into separate databases (e.g., Postgres for relational, DynamoDB/Mongo for document). Use an event-driven approach to maintain eventual consistency."
        ),
        # Template 3: High availability
        lambda c, s, i, comp, t, l: (
            f"Design a highly available architecture for a {i} app on {c} handling {s} users with strict SLAs.",
            f"## High Availability Design on {c}\n\n**1. SLA & Redundancy Requirements:**\nTo achieve 99.99% availability for {s} users, eliminate single points of failure (SPOFs) across all tiers.\n\n**2. Multi-Zone / Multi-Region:**\nDeploy compute instances across at least 3 Availability Zones (AZs). Use an active-active or active-passive multi-region setup with automated DNS failover (e.g., Route53).\n\n**3. Data Tier Resilience:**\nUse a managed multi-AZ database with synchronous replication. For caching, deploy a distributed cache cluster (e.g., Redis) across AZs.\n\n**4. Failover & Monitoring:**\nImplement aggressive health checks and circuit breakers in application logic. Use automated infrastructure-as-code (IaC) to recreate environments during disaster scenarios."
        ),
        # Template 4: Security architecture
        lambda c, s, i, comp, t, l: (
            f"Outline a security architecture for a {comp} compliant {i} workload on {c}.",
            f"## Secure Architecture for {comp} Compliance\n\n**1. Threat Modeling:**\nIdentify attack vectors specific to the {i} industry. Establish a Zero-Trust architecture boundary.\n\n**2. Defense in Depth:**\n- Network: WAF at the edge, private subnets for compute/data, and strict Security Groups.\n- Identity: Centralized IAM with MFA, principle of least privilege, and short-lived credentials.\n- Data: Encryption at rest (KMS) and in transit (TLS 1.2+).\n\n**3. Compliance Controls ({comp}):**\nImplement continuous compliance monitoring (e.g., AWS Config or Azure Policy). Ensure audit logging (CloudTrail) is enabled and shipped to an immutable storage bucket.\n\n**4. Incident Response:**\nAutomated alerting for anomalous behavior. Isolate compromised resources automatically using serverless functions triggered by security events."
        ),
        # Template 5: Data pipeline
        lambda c, s, i, comp, t, l: (
            f"Design a scalable data pipeline on {c} for an {i} business generating {s} events per day.",
            f"## Data Pipeline Architecture\n\n**1. Ingestion Layer:**\nUse a managed messaging queue or streaming platform (e.g., Kafka, Kinesis) to handle high-throughput event ingestion from the {i} apps.\n\n**2. Processing (Stream & Batch):**\nImplement a Lambda/Kappa architecture. Use stream processing (Flink/Spark Streaming) for real-time fraud/anomaly detection, and batch processing (Airflow + Spark) for daily aggregates.\n\n**3. Storage (Data Lakehouse):**\nStore raw events in cheap object storage (S3/GCS) partitioned by date. Use a Data Warehouse (Snowflake/BigQuery) for structured analytics and reporting.\n\n**4. Serving & Observability:**\nProvide data marts for business analysts. Implement strict schema validation at the ingestion point and monitor data freshness and pipeline latency."
        ),
        # Template 6: Event-driven systems
        lambda c, s, i, comp, t, l: (
            f"Explain how to implement an event-driven architecture for a {i} system on {c} handling {s} transactions.",
            f"## Event-Driven Architecture\n\n**1. Event Sources:**\nMicroservices emit domain events (e.g., 'OrderPlaced') rather than calling other services directly. This decouples producers and consumers.\n\n**2. Message Broker:**\nUtilize a scalable broker (e.g., EventBridge, Kafka, or Pub/Sub on {c}). Implement topics/channels based on domain aggregates.\n\n**3. Consumer Groups & Scaling:**\nDeploy consumer services that subscribe to relevant topics. Ensure consumers are idempotent to handle 'at-least-once' delivery semantics safely during network retries.\n\n**4. Choreography vs. Orchestration:**\nUse choreography for simple workflows, but implement a central orchestrator (e.g., Step Functions or Temporal) for complex distributed transactions spanning multiple {i} services."
        ),
        # Template 7: Cost optimization
        lambda c, s, i, comp, t, l: (
            f"How can a {t} optimize their {c} cloud costs for an {i} platform with {s} users?",
            f"## Cloud Cost Optimization Strategy\n\n**1. Spend Analysis & Visibility:**\nImplement strict resource tagging (by environment, service, team). Use {c} cost explorer tools to identify unattached EBS volumes, idle load balancers, and over-provisioned instances.\n\n**2. Right-Sizing Compute:**\nAnalyze CPU/Memory metrics to right-size instances. For a {t}, adopt Serverless (Lambda/CloudRun) where traffic is bursty to scale to zero, and use container orchestration (K8s) for predictable baselines.\n\n**3. Pricing Models:**\nLeverage Spot Instances for stateless, fault-tolerant workloads (e.g., batch processing). Purchase Reserved Instances or Compute Savings Plans for steady-state baseline capacity.\n\n**4. Storage Tiering:**\nSet up lifecycle policies on object storage to move older data to colder, cheaper storage tiers automatically (e.g., S3 Glacier)."
        ),
        # Template 8: Performance engineering
        lambda c, s, i, comp, t, l: (
            f"Design a performance engineering strategy for a high-traffic {i} site on {c} serving {s} users.",
            f"## Performance Engineering & Optimization\n\n**1. Bottleneck Identification:**\nImplement distributed tracing (e.g., OpenTelemetry) to track request latency across microservices. Profile application code to find CPU/Memory hotspots.\n\n**2. Caching Strategy:**\nImplement a multi-tiered caching approach: edge caching (CDN) for static assets, reverse proxy caching for API responses, and application-level caching (Redis/Memcached) for database queries.\n\n**3. Database Optimization:**\nAdd read replicas for read-heavy {i} workloads. Ensure proper indexing on queried columns. Implement connection pooling (e.g., PgBouncer) to prevent database connection exhaustion.\n\n**4. Load Testing:**\nConduct regular load testing using tools like Locust or k6. Simulate {s} user spikes to validate autoscaling policies and ensure graceful degradation under stress."
        ),
    ]
    
    random.seed(42)
    for _ in range(500):
        c = random.choice(cloud_providers)
        s = random.choice(scale_levels)
        i = random.choice(industries)
        comp = random.choice(compliance_standards)
        t = random.choice(team_sizes)
        l = random.choice(legacy_systems)
        
        template = random.choice(templates)
        q, a = template(c, s, i, comp, t, l)
             
        scenarios.append({
            "id": f"arch_syn_{uuid.uuid4().hex[:8]}",
            "text": q,
            "label": a,
            "domain": "architecture"
        })
    return scenarios


def fetch_architecture():
    print("[dataset_loader] === ARCHITECTURE DATASET ===")
    records = []
    
    # 1. Synthetic Expansion
    print("[dataset_loader] [1/2] Generating synthetic architecture templates...")
    syn_records = generate_synthetic_architecture()
    records.extend(syn_records)
    print(f"[dataset_loader]   Synthetic Architecture: {len(syn_records)} records")
        
    # 2. CodeFeedback (Architecture Split with structural curation)
    try:
        print("[dataset_loader] [2/4] Downloading CodeFeedback (Architecture Split)...")
        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train[:30000]")
        added = 0
        coding_keywords = ["debug", "algorithm", "time complexity", "memory", "optimize", "review", "sort", "search"]
        arch_keywords = ["system design", "infrastructure", "scalable", "microservice", "load balancer", "kubernetes", "aws", "docker"]
        
        for item in ds:
            q = item.get("query", "")
            a = item.get("answer", "")
            q_lower = q.lower()
            
            is_coding = any(k in q_lower for k in coding_keywords)
            is_arch = any(k in q_lower for k in arch_keywords)
            
            # Strict router + Curation (Must have formatting and >800 chars)
            if is_arch and not is_coding and len(a) >= 800:
                has_formatting = ("#" in a or "*" in a or "1." in a or "-" in a)
                if has_formatting:
                    records.append({
                        "id": f"arch_cf_{uuid.uuid4().hex[:8]}",
                        "text": q,
                        "label": a,
                        "domain": "architecture"
                    })
                    added += 1
        print(f"[dataset_loader]   CodeFeedback Architecture: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading CodeFeedback: {e}")

    # 3. Shaeh/Software-Architecture (Direct Q&A for systems engineering)
    try:
        print("[dataset_loader] [3/4] Downloading Shaeh/Software-Architecture via raw URL...")
        url = "https://huggingface.co/datasets/Shaeh/Software-Architecture/resolve/main/Software_Architecture_Final.jsonl"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        added = 0
        for line in response.iter_lines():
            if line:
                try:
                    item = json.loads(line.decode('utf-8'))
                    inp = item.get("instruction", "") or item.get("question", "")
                    out = item.get("output", "") or item.get("response", "")
                    if inp and out and len(out) >= 100:
                        records.append({
                            "id": f"arch_shaeh_{uuid.uuid4().hex[:8]}",
                            "text": inp,
                            "label": out,
                            "domain": "architecture"
                        })
                        added += 1
                    if added >= 4000:
                        break
                except Exception:
                    pass
        print(f"[dataset_loader]   Shaeh Software-Architecture: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading Shaeh/Software-Architecture: {e}")

    # 4. yo/overstack (System Design case studies and overengineering critiques)
    try:
        print("[dataset_loader] [4/6] Downloading yo/overstack...")
        ds4 = load_dataset("yo/overstack", split="train[:5000]")
        added = 0
        for item in ds4:
            inp = item.get("input", "")
            verdict = item.get("label", "")
            sol = item.get("appropriate_solution", "")
            expl = item.get("explanation", "")
            
            if inp and sol and expl:
                q = f"Analyze the following architectural scenario:\n{inp}\nAssess whether the solution is appropriate and explain your reasoning."
                a = f"## Assessment\nThe proposed solution is {verdict}.\n\n### Appropriate Solution / Alternatives\n{sol}\n\n### Detailed Explanation\n{expl}"
                
                records.append({
                    "id": f"arch_overstack_{uuid.uuid4().hex[:8]}",
                    "text": q,
                    "label": a,
                    "domain": "architecture"
                })
                added += 1
        print(f"[dataset_loader]   yo/overstack Case Studies: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading yo/overstack: {e}")

    # 5. ajibawa-2023/Software-Architecture (Large-scale architecture Q&A)
    # 450K records available — we pull 15K with strict quality filtering
    existing_hashes = set()
    for r in records:
        h = hashlib.md5((r.get("text", "") + r.get("label", "")).encode()).hexdigest()
        existing_hashes.add(h)

    try:
        print("[dataset_loader] [5/6] Downloading ajibawa-2023/Software-Architecture...")
        ds5 = load_dataset("ajibawa-2023/Software-Architecture", split="train[:40000]")
        added = 0
        target = 10000  # Pull up to 10K quality records

        for item in ds5:
            if added >= target:
                break

            # Handle multiple possible field names
            inp = item.get("instruction", "") or item.get("input", "") or item.get("question", "")
            out = item.get("output", "") or item.get("response", "") or item.get("answer", "")

            if not inp or not out:
                continue

            # Quality gate: answer must be substantial and structured
            if len(out) < 800:
                continue
            has_formatting = ("#" in out or "**" in out or "1." in out or "- " in out)
            if not has_formatting:
                continue

            # Deduplication against existing records
            rec_hash = hashlib.md5((inp + out).encode()).hexdigest()
            if rec_hash in existing_hashes:
                continue
            existing_hashes.add(rec_hash)

            records.append({
                "id": f"arch_ajibawa_{uuid.uuid4().hex[:8]}",
                "text": inp,
                "label": out,
                "domain": "architecture"
            })
            added += 1
        print(f"[dataset_loader]   ajibawa Software-Architecture: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading ajibawa-2023/Software-Architecture: {e}")

    # 6. epinnock/software-architecture-instructions (Targeted arch instructions)
    try:
        print("[dataset_loader] [6/6] Downloading epinnock/software-architecture-instructions...")
        ds6 = load_dataset("epinnock/software-architecture-instructions", split="train")
        added = 0
        for item in ds6:
            inp = item.get("instruction", "") or item.get("input", "") or item.get("question", "")
            out = item.get("output", "") or item.get("response", "") or item.get("answer", "")

            if not inp or not out or len(out) < 200:
                continue

            rec_hash = hashlib.md5((inp + out).encode()).hexdigest()
            if rec_hash in existing_hashes:
                continue
            existing_hashes.add(rec_hash)

            records.append({
                "id": f"arch_epinnock_{uuid.uuid4().hex[:8]}",
                "text": inp,
                "label": out,
                "domain": "architecture"
            })
            added += 1
        print(f"[dataset_loader]   epinnock Architecture Instructions: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading epinnock/software-architecture-instructions: {e}")

    records = _quality_filter(records)
    records = _convert_to_cot(records, fraction=0.30)
    _jsonl_write(records, "data/processed/architecture.jsonl")

def fetch_finance():
    print("\n[dataset_loader] === FINANCE DATASET ===")
    records = []
    
    # 1. gbharti/finance-alpaca
    try:
        print("[dataset_loader] [1/4] Downloading gbharti/finance-alpaca...")
        ds = load_dataset("gbharti/finance-alpaca", split="train[:3000]")
        added = 0
        for item in ds:
            inp = item.get("instruction", "") + " " + item.get("input", "")
            out = item.get("output", "")
            if inp.strip() and out.strip():
                records.append({"id": f"fin_alpaca_{uuid.uuid4().hex[:8]}", "text": inp.strip(), "label": out.strip(), "domain": "finance"})
                added += 1
        print(f"[dataset_loader]   finance-alpaca: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading finance-alpaca: {e}")

    # 2. AdaptLLM/finance-tasks
    try:
        print("[dataset_loader] [2/4] Downloading AdaptLLM/finance-tasks...")
        ds2 = load_dataset("AdaptLLM/finance-tasks", "Headline", split="test[:2000]")
        added = 0
        for item in ds2:
            inp = item.get("input", "")
            options = item.get("options", [])
            gold_idx = item.get("gold_index", -1)
            out = options[gold_idx] if (0 <= gold_idx < len(options)) else ""
            if inp.strip() and out.strip():
                records.append({"id": f"fin_tasks_{uuid.uuid4().hex[:8]}", "text": inp.strip(), "label": out.strip(), "domain": "finance"})
                added += 1
        print(f"[dataset_loader]   finance-tasks: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading finance-tasks: {e}")

    # 3. AdaptLLM/ConvFinQA
    try:
        print("[dataset_loader] [3/4] Downloading AdaptLLM/ConvFinQA...")
        ds3 = load_dataset("AdaptLLM/finance-tasks", "ConvFinQA", split="test[:2000]")
        added = 0
        for item in ds3:
            inp = item.get("input", "")
            out = item.get("label", "")
            if inp.strip() and out.strip():
                records.append({"id": f"fin_conv_{uuid.uuid4().hex[:8]}", "text": inp.strip(), "label": out.strip(), "domain": "finance"})
                added += 1
        print(f"[dataset_loader]   ConvFinQA: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading ConvFinQA: {e}")

    # 4. Josephgflowers/Finance-Instruct-500k (Advanced economics & finance reasoning)
    try:
        print("[dataset_loader] [4/4] Downloading Josephgflowers/Finance-Instruct-500k...")
        ds5 = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train[:10000]")
        added = 0
        for item in ds5:
            inp = item.get("user", "")
            out = item.get("assistant", "")
            if inp.strip() and out.strip():
                records.append({"id": f"fin_instruct_{uuid.uuid4().hex[:8]}", "text": inp.strip(), "label": out.strip(), "domain": "finance"})
                added += 1
        print(f"[dataset_loader]   Finance-Instruct: {added} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading Finance-Instruct-500k: {e}")

    # 5. Synthetic Chart Reading via Permutation
    import itertools
    tickers = ["AAPL", "MSFT", "TSLA", "BTC", "ETH", "SPY"]
    conditions = [
        ("The 50-day moving average crossed below the 200-day moving average. RSI is 25.", "This is a 'Death Cross' (bearish), but the low RSI indicates short-term oversold conditions. Expect a short-term bounce before macro downtrend continuation."),
        ("The MACD line crossed above the signal line. RSI is 55.", "Bullish MACD crossover with neutral-to-bullish RSI. This indicates building upward momentum. Favorable condition for long entries."),
        ("Price broke above the upper Bollinger Band. Volume is extremely high.", "Strong breakout. The high volume confirms the move, though breaking the upper band suggests it may be temporarily overbought and could consolidate."),
        ("The 50-day moving average crossed above the 200-day moving average. RSI is 75.", "This is a 'Golden Cross' (bullish). However, RSI at 75 is overbought, suggesting a potential short-term pullback before continuing the macro uptrend.")
    ]
    synthetic_count = 0
    for tick, (cond_in, cond_out) in itertools.product(tickers, conditions):
        for _ in range(35): # repeat to reach cap
            if synthetic_count >= 800:
                break
            records.append({
                "id": f"fin_synth_{uuid.uuid4().hex[:8]}",
                "text": f"Analyze the chart for {tick}: {cond_in}",
                "label": f"Analysis for {tick}: {cond_out}",
                "domain": "finance"
            })
            synthetic_count += 1
    print(f"[dataset_loader]   Synthetic Chart Reading: {synthetic_count} records")

    records = _quality_filter(records)
    records = _convert_to_cot(records, fraction=0.30)
    records = records[:15000]
    _jsonl_write(records, "data/processed/finance.jsonl")


def fetch_orchestrator():
    print("\n[dataset_loader] === ORCHESTRATOR DATASET ===")
    import random
    records = []
    
    cat_to_domain = {
        "closed_qa": "science",
        "information_extraction": "coding",
        "brainstorming": "architecture",
        "classification": "medical",
        "summarization": "finance",
        "creative_writing": "cyber"
    }

    try:
        print("[dataset_loader] [1/2] Downloading databricks-dolly-15k...")
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        added = 0
        for item in ds:
            cat = item.get("category", "")
            domain = cat_to_domain.get(cat)
            if domain:
                summary = item.get("instruction", "")[:60]
                label_json = json.dumps({"route": [domain], "confidence": 0.95, "multi_domain": False, "query_summary": summary})
                records.append({
                    "id": f"orch_dolly_{uuid.uuid4().hex[:8]}",
                    "text": item.get("instruction", ""),
                    "label": label_json,
                    "domain": "orchestrator"
                })
                added += 1
        print(f"[dataset_loader]   dolly-15k: {added} records mapped")
    except Exception as e:
        print(f"[dataset_loader] Error downloading dolly-15k: {e}")

    try:
        print("[dataset_loader] [2/2] Downloading natural-instructions...")
        ds2 = load_dataset("Muennighoff/natural-instructions", split="train[:10000]")
        added = 0
        for item in ds2:
            task = item.get("task_name", "").lower()
            domain = None
            if "code" in task: domain = "coding"
            elif "med" in task or "health" in task: domain = "medical"
            elif "math" in task or "science" in task: domain = "science"
            elif "fin" in task or "econ" in task: domain = "finance"
            
            if domain:
                inst = item.get("definition", "")
                summary = inst[:60] if inst else "Instruction"
                label_json = json.dumps({"route": [domain], "confidence": 0.95, "multi_domain": False, "query_summary": summary})
                records.append({
                    "id": f"orch_ni_{uuid.uuid4().hex[:8]}",
                    "text": inst,
                    "label": label_json,
                    "domain": "orchestrator"
                })
                added += 1
        print(f"[dataset_loader]   natural-instructions: {added} records mapped")
    except Exception as e:
        print(f"[dataset_loader] Error downloading natural-instructions: {e}")

    # Synthetic Multi-Domain
    domains = ["medical", "cyber", "science", "coding", "architecture", "finance"]
    import itertools
    pairs = list(itertools.combinations(domains, 2))
    added_synth = 0
    for d1, d2 in pairs * 20:
        if added_synth >= 300: break
        text = f"I need a solution involving both {d1} and {d2}."
        label_json = json.dumps({"route": [d1, d2], "confidence": 0.91, "multi_domain": True, "query_summary": text[:60]})
        records.append({
            "id": f"orch_synth_{uuid.uuid4().hex[:8]}",
            "text": text,
            "label": label_json,
            "domain": "orchestrator"
        })
        added_synth += 1
    print(f"[dataset_loader]   Synthetic Multi-Domain: {added_synth} records")

    # Hard-Negative Edge Cases (Domain Disambiguation)
    print("[dataset_loader] [3/3] Generating hard-negative routing examples...")
    hard_negatives = []
    
    # 1. Medical words in non-medical context -> finance/cyber/coding/arch
    hn_med = [
        ("What's the prognosis for this company's financial health?", "finance"),
        ("The network has a virus, how do I quarantine it?", "cyber"),
        ("Diagnose the root cause of this system failure", "architecture"),
        ("This code has a memory leak, what's the treatment?", "coding"),
        ("The patient data pipeline needs a health check", "architecture")
    ] * 20
    
    # 2. Security words in non-security context
    hn_sec = [
        ("How do I inject CSS into this React component?", "coding"),
        ("What's the attack surface of this investment strategy?", "finance"),
        ("Explain SQL injection prevention in my Flask app", "coding"),
        ("Firewall this section of the codebase from external dependencies", "architecture"),
        ("Penetration testing of this algorithm's edge cases", "coding")
    ] * 20
    
    # 3. Multi-domain
    hn_multi = [
        ("Design a HIPAA-compliant patient data architecture", ["architecture", "medical"]),
        ("Build a fraud detection pipeline for crypto transactions", ["finance", "cyber", "coding"]),
        ("What are the security implications of microservice authentication?", ["cyber", "architecture"]),
        ("Develop a clinical trial data analysis platform", ["medical", "science", "coding"])
    ] * 25
    
    # 4. Ambiguous
    hn_amb = [
        ("How does it work?", "clarification_needed"),
        ("Fix this", "clarification_needed"),
        ("Tell me about security", "clarification_needed"),
        ("What's the best approach?", "clarification_needed")
    ] * 25
    
    # 5. Out of domain
    hn_ood = [
        ("Write me a poem about the sunset", "no_domain"),
        ("What's the capital of France?", "no_domain"),
        ("Recommend a good restaurant in Mumbai", "no_domain")
    ] * 30
    
    for text, route in (hn_med + hn_sec + hn_multi + hn_amb + hn_ood):
        route_list = route if isinstance(route, list) else [route]
        is_multi = len(route_list) > 1
        label_json = json.dumps({"route": route_list, "confidence": 0.95, "multi_domain": is_multi, "query_summary": text[:60]})
        hard_negatives.append({
            "id": f"orch_hn_{uuid.uuid4().hex[:8]}",
            "text": text,
            "label": label_json,
            "domain": "orchestrator",
            "is_hard_negative": True
        })
    records.extend(hard_negatives)
    print(f"[dataset_loader]   Hard-Negative Routing: {len(hard_negatives)} records generated")

    # 6. Additional diverse single-domain/technical routing scenarios
    print("[dataset_loader] [4/4] Generating 500 diverse single-domain routing records...")
    synth_routing = []
    
    routing_templates = {
        "medical": [
            "What is the mechanism of action of {med} in treating {cond}?",
            "Detail the clinical trial phase outcomes for the new {med} inhibitor.",
            "A patient presents with acute {cond} and {symptom}. What is the differential diagnosis?",
            "Explain the pharmacokinetics of {med} in renal-impaired patients."
        ],
        "cyber": [
            "How does one analyze a heap dump for a suspected execution of {malware}?",
            "Provide the Yara rule syntax to detect {malware} payload indicators.",
            "Explain how to configure iptables to mitigate a SYN flood attack on port {port}.",
            "What indicators of compromise (IoCs) are associated with the {malware} threat actor group?"
        ],
        "science": [
            "Derive the Schrödinger equation for a particle in a {dim} potential well.",
            "Explain the thermodynamic cycle efficiency of a {cycle} engine under high pressure.",
            "What is the chemical reaction pathway for the synthesis of {compound}?",
            "Describe the mitotic spindle assembly checkpoint mechanism during cellular division."
        ],
        "coding": [
            "Write a Python script using asyncio to concurrently poll {count} endpoints.",
            "How do I override the default serializer behaviour in Django Rest Framework for nested fields?",
            "Refactor this recursive algorithm to use dynamic programming and reduce space complexity.",
            "Explain the difference between a prototype-based inheritance and class-based inheritance in JS."
        ],
        "architecture": [
            "Design a multi-tenant database partitioning scheme for SaaS applications scaling to {count} users.",
            "How do we configure a distributed lock manager using Redis Redlock for distributed consistency?",
            "Outline the load balancer topology required to terminate TLS and route gRPC traffic.",
            "Explain the trade-offs of choosing event sourcing over a traditional CRUD architecture."
        ],
        "finance": [
            "Calculate the Black-Scholes options pricing model variables for a call option valued at {val}.",
            "How does the yield curve inversion predict long-term macroeconomic recession trends?",
            "What is the impact of international double taxation treaties on corporate tax liability?",
            "Outline a portfolio risk management model using Monte Carlo simulations for {val} assets."
        ]
    }
    
    meds = ["metformin", "lisinopril", "atorvastatin", "albuterol", "ibuprofen"]
    conds = ["type 2 diabetes", "hypertension", "hypercholesterolemia", "asthma", "inflammation"]
    symptoms = ["shortness of breath", "chest pain", "severe headache", "nausea", "dizziness"]
    malware = ["Cobalt Strike", "Emotet", "Mimikatz", "WannaCry", "Log4j exploit"]
    ports = ["80", "443", "22", "3389", "8080"]
    dims = ["one-dimensional", "three-dimensional", "spherical", "cylindrical"]
    cycles = ["Carnot", "Rankine", "Otto", "Diesel"]
    compounds = ["aspirin", "paracetamol", "caffeine", "ethanol", "benzene"]
    counts = ["10k", "50k", "100k", "1m", "10m"]
    vals = ["$10M", "$100M", "$1B", "10,000 contracts"]
    
    random.seed(101)
    for domain, prompts in routing_templates.items():
        for _ in range(85): # ~510 total records
            prompt_template = random.choice(prompts)
            text = prompt_template.format(
                med=random.choice(meds), cond=random.choice(conds), symptom=random.choice(symptoms),
                malware=random.choice(malware), port=random.choice(ports), dim=random.choice(dims),
                cycle=random.choice(cycles), compound=random.choice(compounds), count=random.choice(counts),
                val=random.choice(vals)
            )
            label_json = json.dumps({"route": [domain], "confidence": 0.95, "multi_domain": False, "query_summary": text[:60]})
            synth_routing.append({
                "id": f"orch_synth_route_{uuid.uuid4().hex[:8]}",
                "text": text,
                "label": label_json,
                "domain": "orchestrator"
            })
    records.extend(synth_routing)
    print(f"[dataset_loader]   Diverse Single-Domain Routing: {len(synth_routing)} records generated")

    # 7. System Architecture & Implementation multi-domain records
    print("[dataset_loader] [5/5] Generating system design/architecture expertise records...")
    system_routing = []
    sys_templates = [
        ("Build a {domain_noun} prediction platform with high scalability.", ["{domain}", "coding", "architecture"]),
        ("Design an automated {domain_noun} analysis pipeline using real-time sensors.", ["{domain}", "coding", "architecture"]),
        ("Develop a secure cloud-native application for {domain_noun} monitoring.", ["{domain}", "coding", "architecture"]),
        ("Create an AI-driven system to optimize {domain_noun} processes.", ["{domain}", "coding", "architecture"]),
        ("Implement a distributed database system for storing {domain_noun} logs.", ["{domain}", "coding", "architecture"]),
        ("We need to design a software platform for {domain_noun} simulation.", ["{domain}", "coding", "architecture"]),
        ("Help me build the backend and infrastructure for a {domain_noun} service.", ["{domain}", "coding", "architecture"]),
        ("Redesign our {domain_noun} pipeline to handle 100k requests per second safely.", ["{domain}", "coding", "architecture"])
    ]
    domain_mappings = {
        "finance": ["stock market", "high frequency trading", "quantitative investment", "portfolio risk", "cryptocurrency trading"],
        "medical": ["patient EHR data", "clinical trial telemetry", "remote patient monitoring", "hospital management", "medical billing"],
        "cyber": ["threat detection", "malware analysis", "network security monitoring", "access control and auth", "vulnerability scanning"],
        "science": ["climate simulation", "genomic sequencing", "chemical reaction pathways", "predictive maintenance", "particle physics modeling"]
    }
    
    for domain, nouns in domain_mappings.items():
        for noun in nouns:
            for template, route_template in sys_templates:
                text = template.format(domain_noun=noun)
                route = [r.format(domain=domain) for r in route_template]
                label_json = json.dumps({"route": route, "confidence": 0.95, "multi_domain": True, "query_summary": text[:60]})
                system_routing.append({
                    "id": f"orch_sys_route_{uuid.uuid4().hex[:8]}",
                    "text": text,
                    "label": label_json,
                    "domain": "orchestrator"
                })
    records.extend(system_routing)
    print(f"[dataset_loader]   System Architecture & Implementation: {len(system_routing)} records generated")

    records = _quality_filter(records, min_label_len=15)
    random.shuffle(records)
    records = records[:5500]
    _jsonl_write(records, "data/processed/orchestrator.jsonl")


def fetch_meta_reasoner():
    print("\n[dataset_loader] === META-REASONER DATASET ===")
    records = []
    
    try:
        print("[dataset_loader] [1/1] Downloading openbmb/UltraFeedback...")
        ds = load_dataset("openbmb/UltraFeedback", split="train[:15000]")
        pass_count = 0
        fail_count = 0
        for item in ds:
            comps = item.get("completions", [])
            inst = item.get("instruction", "")
            if not comps or len(comps) < 2 or not inst:
                continue
            
            try:
                comps_scored = []
                for c in comps:
                    score = 0
                    if "annotations" in c and c["annotations"] and "overall_score" in c["annotations"]:
                        score = float(c["annotations"]["overall_score"])
                    elif "rating" in c:
                        score = float(c["rating"])
                    comps_scored.append((score, c.get("response", "")))
                comps_scored.sort(key=lambda x: x[0], reverse=True)
                high_resp = comps_scored[0][1]
                low_resp = comps_scored[-1][1]
                
                if high_resp and low_resp and high_resp != low_resp:
                    # Target ratio: 60% pass, 40% fail
                    if pass_count < 2820 and (pass_count * 4 <= fail_count * 6 or fail_count >= 1880):
                        val = {"input": f"{inst}\n\n{high_resp}", "verdict": "pass", "corrected_output": high_resp}
                        records.append({"id": f"meta_uf_p_{uuid.uuid4().hex[:8]}", "text": val["input"], "label": json.dumps(val), "domain": "meta_reasoner"})
                        pass_count += 1
                    elif fail_count < 1880:
                        val = {"input": f"{inst}\n\n{low_resp}", "verdict": "fail", "corrected_output": high_resp}
                        records.append({"id": f"meta_uf_f_{uuid.uuid4().hex[:8]}", "text": val["input"], "label": json.dumps(val), "domain": "meta_reasoner"})
                        fail_count += 1
            except Exception:
                pass
            if pass_count >= 2820 and fail_count >= 1880:
                break
        print(f"[dataset_loader]   UltraFeedback: {pass_count + fail_count} records")
    except Exception as e:
        print(f"[dataset_loader] Error downloading UltraFeedback: {e}")

    # Synthetic Contradictions
    added_synth = 0
    domains = ["medical", "cyber", "science", "coding", "architecture", "finance"]
    import itertools
    for d1, d2 in itertools.combinations(domains, 2):
        for i in range(20):
            if added_synth >= 300: break
            inst = f"How do I securely deploy a {d1} application?"
            bad = f"[Specialist: {d1}] You should roll your own custom cryptographic protocols for ultimate security."
            good = f"[Specialist: {d2}] Never roll your own crypto. Use standard established libraries."
            val = {"input": f"{inst}\n\n{bad}\n\n{good}", "verdict": "fail", "corrected_output": f"The {d1} specialist is incorrect. The {d2} specialist is correct: never roll your own crypto. Use standard libraries."}
            records.append({"id": f"meta_synth_{uuid.uuid4().hex[:8]}", "text": val["input"], "label": json.dumps(val), "domain": "meta_reasoner"})
            added_synth += 1
    print(f"[dataset_loader]   Synthetic Contradictions: {added_synth} records")

    # Synthetic Conflicting Specialist Scenarios (+500 records)
    print("[dataset_loader] Generating 500 synthetic conflicting specialist scenarios...")
    added_conflict = 0
    conflict_scenarios = [
        # Medical / Science conflict
        ("A patient shows symptoms of elevated enzyme levels. Specialist 1 suggests instant surgical excision. Specialist 2 argues for non-invasive metabolic diagnostic checks first.",
         "The recommendation from Specialist 2 is correct. Surgical intervention without preliminary non-invasive diagnostic confirmation represents an unnecessary clinical risk."),
        # Cyber / Architecture conflict
        ("How should we secure microservices communication? Specialist 1 recommends leaving internal networks unencrypted for latency optimization. Specialist 2 insists on Mutual TLS (mTLS).",
         "Specialist 2 is correct. Latency considerations do not justify leaving internal microservice communications unencrypted. Mutual TLS (mTLS) is standard zero-trust architecture practice."),
        # Finance / Coding conflict
        ("How to implement an automated stock arbitrage algorithm? Specialist 1 suggests running floating-point arithmetic for price precision. Specialist 2 recommends decimal/fixed-point arithmetic to avoid rounding bugs.",
         "Specialist 2 is correct. Floating-point arithmetic introduces rounding inaccuracies that degrade financial transaction systems. Fixed-point or Decimal types must be used."),
        # Coding / Architecture conflict
        ("How to handle distributed transaction failures? Specialist 1 suggests implementing a basic HTTP retry loop. Specialist 2 recommends implementing the Saga pattern with rollback compensating transactions.",
         "Specialist 2 is correct. HTTP retries alone cannot maintain eventual consistency across distributed datastores during network failures. The Saga pattern provides robust rollback logic.")
    ]
    
    for prompt, resolution in conflict_scenarios * 125: # ~500 records
        val = {
            "input": f"Resolve the following specialist disagreement:\n{prompt}",
            "verdict": "conflict_resolved",
            "corrected_output": resolution
        }
        records.append({
            "id": f"meta_conflict_{uuid.uuid4().hex[:8]}",
            "text": val["input"],
            "label": json.dumps(val),
            "domain": "meta_reasoner"
        })
        added_conflict += 1
    print(f"[dataset_loader]   Synthetic Conflicts: {added_conflict} records generated")

    # Synthetic CoT Chain Evaluation Scenarios (+500 records)
    print("[dataset_loader] Generating 500 synthetic CoT chain evaluation records...")
    added_cot_eval = 0
    cot_eval_scenarios = [
        ("Chain A steps: [1: Identify symptom] -> [2: Diagnose without tests] -> [3: Conclude treatment]. Chain B steps: [1: Identify symptom] -> [2: Analyze clinical lab results] -> [3: Formulate differential diagnosis] -> [4: Conclude target treatment]. Which chain is logically valid?",
         "Chain B is logically valid. Chain A contains a logical gap between steps 1 and 2 by attempting to diagnose without evaluating tests."),
        ("Chain A steps: [1: Parse server log] -> [2: Note suspicious payload IP] -> [3: Conclude zero malicious activity]. Chain B steps: [1: Parse server log] -> [2: Note suspicious payload IP] -> [3: Cross-reference threat intelligence database] -> [4: Conclude block traffic from IP]. Which chain is logically valid?",
         "Chain B is logically valid. Chain A draws a conclusion in Step 3 that contradicts the evidence noted in Step 2 without performing any verification or cross-referencing.")
    ]
    
    for prompt, resolution in cot_eval_scenarios * 250: # ~500 records
        val = {
            "input": f"Analyze these reasoning chains and select the correct logical approach:\n{prompt}",
            "verdict": "cot_evaluated",
            "corrected_output": resolution
        }
        records.append({
            "id": f"meta_cot_eval_{uuid.uuid4().hex[:8]}",
            "text": val["input"],
            "label": json.dumps(val),
            "domain": "meta_reasoner"
        })
        added_cot_eval += 1
    print(f"[dataset_loader]   Synthetic CoT Eval: {added_cot_eval} records generated")

    records = _quality_filter(records)
    records = _convert_to_cot(records, fraction=0.30)
    records = records[:6000]
    _jsonl_write(records, "data/processed/meta_reasoner.jsonl")


def audit_specialist_overlap():
    """Checks for hash-based deduplication across coding and architecture datasets."""
    print("\n[dataset_loader] === AUDITING SPECIALIST OVERLAP ===")
    
    def get_hashes(filepath: str) -> set:
        import hashlib
        hashes = set()
        if not os.path.exists(filepath):
            return hashes
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    text = rec.get("text", "").strip()
                    if text:
                        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        hashes.add(h)
        return hashes

    coding_hashes = get_hashes("data/processed/coding.jsonl")
    arch_hashes = get_hashes("data/processed/architecture.jsonl")
    
    overlap = coding_hashes.intersection(arch_hashes)
    
    print(f"Coding records:       {len(coding_hashes)}")
    print(f"Architecture records: {len(arch_hashes)}")
    print(f"Overlap detected:     {len(overlap)}")
    
    if len(overlap) > 0:
        print("[!] ERROR: Router failure detected. Datasets overlap.")
        sys.exit(1)
    else:
        print("[+] SUCCESS: Zero overlap between Coding and Architecture.")


if __name__ == "__main__":
    import sys
    import os
    import json
    import io
    import re
    
    class TeeLogger(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
            
        def flush(self):
            for f in self.files:
                f.flush()

    buffer = io.StringIO()
    sys.stdout = TeeLogger(sys.stdout, buffer)

    print("[dataset_loader] Starting automated extraction from HuggingFace (v2)...")
    if os.path.exists("data/processed/dataset_manifest.json"):
        print("[dataset_loader] Found existing dataset_manifest.json. Bypassing dataset generation.")
        print("[dataset_loader] Extraction complete! Ready for training.")
        sys.exit(0)

    os.makedirs("data/processed", exist_ok=True)
    fetch_cyber()
    # fetch_science()  # Already trained
    fetch_coding()
    fetch_architecture()
    # fetch_finance()  # Already trained
    fetch_orchestrator()
    # fetch_meta_reasoner()  # Already trained
    
    # Step 4: Overlap Audit
    audit_specialist_overlap()
    
    print("[dataset_loader] Generating dataset_manifest.json...")
    manifest = {}
    current_domain = None
    
    for line in buffer.getvalue().split('\n'):
        if "=== " in line and "DATASET" in line:
            current_domain = line.split("===")[1].replace("DATASET", "").replace("(v2 — Expanded)", "").strip().lower()
            manifest[current_domain] = {}
        elif "[dataset_loader]   " in line and ":" in line and " records" in line:
            match = re.search(r'\[dataset_loader\]\s+(.*?):\s+(\d+)\s+records', line)
            if match and current_domain:
                source = match.group(1).strip()
                count = int(match.group(2))
                manifest[current_domain][source] = count

    with open("data/processed/dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        
    print("[dataset_loader] Extraction complete! Ready for training.")


