# ATS Resume Tailoring Agent

An autonomous, agentic GenAI system that intelligently tailors LaTeX resumes to any Job Description (JD) — reducing manual editing time by ~80% while preserving factual accuracy, structure, and honesty.

### Why this exists
Job applications are painful. Most people spend hours rewriting resumes to match JDs, fighting ATS filters, and risking over-inflation of titles/experience.  
This agent automates the hard parts: keyword extraction, natural rewriting, structure preservation, LaTeX safety, and compilation — all in a self-healing, production-oriented loop.

### How it works (high-level)
1. **Upload** your LaTeX resume code + target Job Description  
2. **KeyBERT** (local) extracts the 15 most important technical keywords from the JD  
3. **Groq + Llama 3.3** (ultra-fast) rewrites bullet points using a strict "Technical Editor" persona that never inflates titles, never adds unrelated boilerplate, and only integrates keywords naturally  
4. **Self-healing compilation loop** using local MiKTeX: automatically fixes LaTeX syntax errors with up to 3 retries by feeding error logs back to the LLM  
5. **Semantic alignment scoring** via Sentence-Transformers (regex-cleaned content) measures how well the new resume matches the JD  
6. Download your perfectly compiled, tailored PDF + match score

### Key Technical Features
- **Agentic architecture** powered by **LangGraph** state machine (dynamic routing, self-correction loops)  
- **Cost-aware keyword extraction** with **KeyBERT** — avoids expensive LLM calls on full JDs  
- **Persona integrity guardrails** — strictly prevents title inflation (e.g., "Intern" never becomes "Senior")  
- **Logical consistency rules** — skips irrelevant JD phrases (e.g., no "health optimization" in AI projects)  
- **Self-healing LaTeX compilation** — parses pdflatex errors and feeds them back for automatic fixes  
- **Factual drift prevention** — enforces high similarity between original and rewritten content  
- **Production-ready backend** — FastAPI API endpoints (`/tailor`, `/download`)  
- **Local-first deployment** — uses MiKTeX + uv (no Docker hell — build time reduced from 2+ hours to ~2 minutes)  
- **Visualization** — interactive LangGraph diagram generated via mermaid.ink in Jupyter Notebook  

### Current Status
**Still in active development**  
- Core pipeline is fully functional and already produces dramatically better resumes than manual keyword stuffing  
- Ongoing improvements: better keyword filtering, stronger hallucination guards, user feedback loop, and optional fine-tuning of the LLM  
- Expect frequent updates — feedback/PRs very welcome!

### Tech Stack
- Python · LangGraph · LangChain  
- Groq (Llama 3.3-70b)  
- KeyBERT · Sentence-Transformers  
- FastAPI · MiKTeX (local LaTeX)  
- uv (for fast local environment)  
- Optional: Docker (previous attempt — abandoned for speed)

### How to Run Locally (Coming Soon)
Detailed setup instructions will be added shortly.  
Current requirements:  
- Python 3.10+  
- MiKTeX installed (for pdflatex)  
- Groq API key (set in `.env`)

### Contributing
This project is open to contributions!  
Especially interested in:
- Better prompt engineering for even cleaner rewrites
- Additional guardrails / validation layers
- UI improvements (Streamlit frontend?)
- Deployment options (Render, Railway, etc.)

Feel free to open issues or PRs.

### License
MIT License

**Happy tailoring — and good luck with your applications!** 🚀

---

**Still in active development** — expect breaking changes, new features, and improved quality over the coming weeks/months.
