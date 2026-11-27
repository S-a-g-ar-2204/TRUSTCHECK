<p align="center"><img src="assets/logo.png" width="180" alt="TrustCheck logo"></p>

# TrustCheck - AI Chatbox Vulnerability Tester

Streamlit dashboard that red-teams an AI chatbot and analyses every reply with a **built-in AI Judge Engine** (no external judge model needed).

## Features
- Configure only the chatbot under test: OpenAI, Claude, Gemini, Groq, any OpenAI-compatible API (cloud or local server), or a built-in **Demo** bot
- 23 attacks: Prompt Injection, Jailbreak, System Prompt Extraction, Sensitive Data Leakage, Harmful Content Generation (+ custom attacks)
- Built-in Judge: Successful/Failed verdict, severity (Critical/High/Medium/Low/Safe), explanation and detection signals
- Optional AI second opinion using the same target model
- Chat tab: talk to the target chatbot directly, send attacks from the library, see judge verdicts and secret-leak flags
- Interactive dashboard: live attack feed, quick scan, radar chart, severity filters + search, cards/table view, per-attack retest
- "What leaked" panel: every leaked value (secrets, prompt lines, personal data) is listed in full and highlighted inside the reply, with a scan-wide summary
- Defence presets (Weak / Balanced / Strict / Custom) and a run-comparison chart to prove a defence works
- Overall risk score (0-100), colour indicators, HTML/JSON/CSV reports

## Project files
- `ai_vuln_tester.py`: the app
- `assets/logo.png`: logo used in the app (optional, the app runs without it)
- `.streamlit/config.toml`: dark red theme
- `requirements.txt`

## Run
```
pip install -r requirements.txt
python -m streamlit run ai_vuln_tester.py
```
Choose **Demo** in the sidebar to preview the dashboard without any model or API key.

## Responsible use
Only test models and applications you own or are explicitly authorised to test.
