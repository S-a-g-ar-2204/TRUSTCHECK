# AI Chatbox Vulnerability Tester

Streamlit dashboard that red-teams an LLM chatbot (local or cloud) and uses a second "Judge" LLM to score each attack.

## Features
- Targets: Ollama, LM Studio, OpenAI, Claude, Gemini, Groq, any OpenAI-compatible API
- Tests: Prompt Injection, Jailbreak, System Prompt Extraction, Sensitive Data Leakage, Harmful Content Generation
- AI Judge: Successful/Failed verdict, severity (Critical/High/Medium/Low/Safe), explanation
- Canary-based automatic leak detection, overall risk score (0-100), HTML/JSON/CSV reports

## Run
```
pip install -r requirements.txt
streamlit run ai_vuln_tester.py
```

API keys: enter in the UI or set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`.

## Responsible use
Only test models and applications you own or are explicitly authorised to test.
