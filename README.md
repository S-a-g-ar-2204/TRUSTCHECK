<p align="center"><img src="assets/logo.png" width="180" alt="TrustCheck logo"></p>

# TrustCheck - AI Chatbox Vulnerability Tester

Streamlit dashboard that red-teams an AI chatbot and analyses every reply with a **built-in AI Judge Engine** (no external judge model needed).

TrustCheck tests the model **exactly as you connect it**: no hidden persona, no injected system prompt. Every attack can be repeated by hand and gives the same reply.

## Features
- Connect only the model under test: a local server (Ollama, LM Studio, ...), OpenAI, Claude, Gemini, Groq, any OpenAI-compatible API, or a built-in **Demo** bot
- 23 attacks: Prompt Injection, Jailbreak, System Prompt Extraction, Sensitive Data Leakage, Harmful Content Generation (+ custom attacks)
- Built-in Judge: Successful/Failed verdict, severity (Critical/High/Medium/Low/Safe), explanation and detection signals
- Optional AI second opinion using the same target model
- **Match manual testing** (on by default): temperature 0 and a fixed seed, so the tool and a manual run give the same text
- Optional system prompt and secrets to watch for, for chatbots that are deployed with one (empty by default)
- Chat tab: talk to the model directly with your logo as its avatar, send attacks from the library, see judge verdicts
- Interactive dashboard: live attack feed, quick scan, radar chart, severity filters + search, cards/table view, per-attack retest
- "What leaked" panel and a run-comparison chart
- Overall risk score (0-100), colour indicators, HTML/JSON/CSV reports

## Project files
- `ai_vuln_tester.py`: the app
- `assets/logo.png`: logo (hero, sidebar, report and chat avatar). Also found if placed next to the script
- `.streamlit/config.toml`: dark red theme
- `requirements.txt`

## Run
```
pip install -r requirements.txt
python -m streamlit run ai_vuln_tester.py
```
Choose **Demo** in the sidebar to preview the dashboard without any model or API key.

## Repeating a test by hand (Ollama example)
Open the **Reproduce a reply by hand** panel in the Chat tab. It prints the exact commands, for example:
```
ollama run llama3.2:1b
/set parameter temperature 0
/set parameter num_predict 600
/set parameter seed 42
```
Then paste the same attack text as your first message (use `/clear` between attacks).

## Notes on the judge
- The System Prompt Extraction category needs a system prompt, so it is switched off until you add one under *Test settings*.
- Without a configured secret, a credential-looking string in a reply is treated as *fabricated*, not leaked, and is rated at most Medium.
- The judge is rule-based. Always review "Successful" and borderline results yourself.

## Responsible use
Only test models and applications you own or are explicitly authorised to test.
