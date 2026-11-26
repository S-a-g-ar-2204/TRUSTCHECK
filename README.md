# TrustCheck - AI Chatbox Vulnerability Tester

Streamlit dashboard that red-teams an AI chatbot and analyses every reply with a **built-in AI Judge Engine** (no external judge model needed).

## Features
- Configure only the chatbot under test: Ollama, LM Studio, OpenAI, Claude, Gemini, Groq, any OpenAI-compatible API, or a built-in **Demo** bot
- 23 attacks: Prompt Injection, Jailbreak, System Prompt Extraction, Sensitive Data Leakage, Harmful Content Generation (+ custom attacks)
- Built-in Judge: Successful/Failed verdict, severity (Critical/High/Medium/Low/Safe), explanation and detection signals
- Optional AI second opinion using the same target model
- Overall risk score (0-100), colour indicators, HTML/JSON/CSV reports

## Run
```
pip install -r requirements.txt
python -m streamlit run ai_vuln_tester.py
```
Choose **Demo** in the sidebar to preview the dashboard without any model or API key.

## Responsible use
Only test models and applications you own or are explicitly authorised to test.
