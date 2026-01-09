import os
import subprocess
import re
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from .state import ResumeState

# Lazy-loaded models to speed up startup/visualization
_kw_model = None
_score_model = None
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile")
    return _llm

def get_kw_model():
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT()
    return _kw_model

def get_score_model():
    global _score_model
    if _score_model is None:
        _score_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _score_model

def extract_keywords_node(state: ResumeState):
    """Extract top keywords from the JD using KeyBERT."""
    kw_model = get_kw_model()
    keywords = kw_model.extract_keywords(
        state['job_description'], 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=15
    )
    extracted = [k[0] for k in keywords]
    return {"extracted_keywords": extracted}

def filter_keywords_node(state: ResumeState):
    """Refine keywords using LLM to remove non-technical boilerplate."""
    system_prompt = (
        "You are a technical recruiter. Your task is to filter a list of keywords extracted from a Job Description. "
        "Keep ONLY technical skills, tools, frameworks, and core engineering concepts. "
        "Discard company names, mission statements, and vague boilerplate (e.g., 'health optimization', 'care-aided', 'equal opportunity'). "
        "Return ONLY a comma-separated list of the best 10 technical keywords. If none, return 'technical skills'."
    )
    
    user_prompt = f"Raw Keywords: {', '.join(state['extracted_keywords'])}\n\nTechnical Keywords:"
    
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    filtered = [k.strip() for k in response.content.split(",")]
    return {"filtered_keywords": filtered[:10]}

def rewrite_resume_node(state: ResumeState):
    """Rewrite sections of the LaTeX resume to incorporate keywords."""
    keywords_str = ", ".join(state['filtered_keywords'])
    
    system_prompt = (
        "You are an expert technical resume writer. Your goal is to tailor a resume to a Job Description while preserving the EXACT LaTeX structure and facts. "
        "UNBREAKABLE RULES:\n"
        "1. **STRICT PRESERVATION**: Do NOT change your name, contact info, job titles, companies, or dates. An 'Intern' MUST stay an 'Intern'.\n"
        "2. **TECHNICAL ACCURACY**: Only integrate keywords if they fit LOGICALLY into your actual experience. Do NOT add skills you don't have.\n"
        "3. **BOILERPLATE EXCLUSION**: Never inject company mission statements or non-technical slogans (e.g., 'health optimization').\n"
        "4. **LaTeX INTEGRITY**: Do NOT change the preamble, geometry, or styling. Only rewrite content within blocks like Summary, Experience, and Projects.\n"
        "5. **SINGLE PAGE**: Keep descriptions concise. Do NOT push the resume to a second page.\n"
        "6. **NO HALLUCinations**: Do NOT invent new responsibilities. Only rephrase existing ones to highlight relevant technical keywords.\n"
        "7. Output ONLY the complete, updated LaTeX code block."
    )
    
    user_prompt = (
        f"Filtered Keywords: {keywords_str}\n\n"
        f"Original LaTeX Resume:\n{state['resume_latex']}\n\n"
        "Tailor the resume naturally. Return ONLY the LaTeX code."
    )
    
    if state.get('compilation_error'):
        user_prompt += f"\n\nPrevious compilation error to fix: {state['compilation_error']}"
    
    if state.get('validation_reason'):
        user_prompt += f"\n\nContent validation feedback to fix: {state['validation_reason']}"

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    # Clean up response (remove markdown code blocks if present)
    new_latex = response.content.strip()
    if new_latex.startswith("```"):
        new_latex = "\n".join(new_latex.split("\n")[1:-1])
        
    return {"resume_latex": new_latex, "iteration_count": state.get('iteration_count', 0) + 1}

def validate_content_node(state: ResumeState):
    """Check for hallucinations or title changes in the tailored resume."""
    system_prompt = (
        "You are an audit agent. Compare the tailored LaTeX resume to the original facts. "
        "Check for:\n"
        "1. Title Inflation (e.g., Intern changed to Senior).\n"
        "2. Hallucinations (e.g., Healthcare mission statements in an AI project).\n"
        "3. Fact Changes (e.g., Changing dates or companies).\n"
        "If the resume IS FAITHFUL, return 'PASS'.\n"
        "If there are issues, return 'FAIL: [brief reason]'."
    )
    
    user_prompt = (
        f"Original Resume:\n{state['original_resume_latex']}\n\n"
        f"Tailored Resume:\n{state['resume_latex']}\n\n"
        "Audit the tailored resume:"
    )
    
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    if "PASS" in response.content.upper():
        return {"validation_reason": None}
    else:
        return {"validation_reason": response.content.strip()}

def compile_node(state: ResumeState):
    """Attempt to compile the LaTeX resume."""
    temp_dir = "temp_latex"
    os.makedirs(temp_dir, exist_ok=True)
    tex_path = os.path.join(temp_dir, "resume.tex")
    
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(state['resume_latex'])

    # Run pdflatex
    try:
        # Use absolute path if provided in .env, otherwise fallback to system 'pdflatex'
        pdflatex_cmd = os.getenv("PDFLATEX_PATH", "pdflatex")
        
        # Clean path
        pdflatex_cmd = pdflatex_cmd.strip('"').strip("'")
        pdflatex_cmd = os.path.normpath(pdflatex_cmd)
        
        if pdflatex_cmd != "pdflatex" and not os.path.exists(pdflatex_cmd):
            error_msg = f"LaTeX binary not found at specified path: {pdflatex_cmd}"
            print(error_msg)
            return {"compilation_error": error_msg}

        print(f"--- Attempting LaTeX compilation for: {tex_path} using {pdflatex_cmd} ---")
        
        # Use list format for better reliability on Windows without shell=True if possible
        # but pdflatex sometimes needs a shell to resolve dependencies. 
        # We'll use list format and remove shell=True to be safer.
        result = subprocess.run(
            [pdflatex_cmd, "-interaction=nonstopmode", "-output-directory", temp_dir, "resume.tex"],
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode == 0:
            pdf_path = os.path.join(temp_dir, "resume.pdf")
            if os.path.exists(pdf_path):
                print(f"Success: PDF generated at {pdf_path}")
                return {"pdf_path": pdf_path, "compilation_error": None}
            else:
                print("Error: pdflatex returned 0 but pdf file was not found.")
                return {"compilation_error": "pdflatex finished but no PDF was found."}
        else:
            # Combine stdout and stderr for better debugging
            error_msg = (result.stdout + "\n" + result.stderr).strip()
            print(f"Compilation failed with exit code {result.returncode}")
            print(f"Error details: {error_msg[:200]}")
            return {"compilation_error": error_msg or "Unknown LaTeX error"}
    except Exception as e:
        print(f"Subprocess exception: {str(e)}")
        return {"compilation_error": str(e)}

def score_node(state: ResumeState):
    """Calculate semantic similarity between JD and tailored resume."""
    # Strip basic LaTeX tags to compare actual content
    def clean_latex(text):
        text = re.sub(r'\\[a-zA-Z]+(\{.*?\})?', '', text) # Remove commands
        text = re.sub(r'\{|\}', '', text) # Remove braces
        text = re.sub(r'%.*?\n', '', text) # Remove comments
        return text.strip()

    clean_resume = clean_latex(state['resume_latex'])
    clean_original = clean_latex(state['original_resume_latex'])
    
    score_model = get_score_model()
    
    # JD Match Score
    emb_jd = score_model.encode(state['job_description'])
    emb_res = score_model.encode(clean_resume)
    jd_score = util.cos_sim(emb_jd, emb_res).item()
    
    # Factual Drift Score
    emb_orig = score_model.encode(clean_original)
    drift_score = util.cos_sim(emb_orig, emb_res).item()
    
    return {
        "semantic_score": jd_score * 100,
        "drift_score": drift_score * 100
    }
