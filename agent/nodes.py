import re
import os
import subprocess
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .state import ResumeState

# Lazy-loaded models to speed up startup/visualization
_kw_model = None
_score_model = None
_llm = None

def escape_latex_content(text):
    """
    Surgically escape special characters while protecting LaTeX logic and comments.
    Ensures % at the start of a line remains a comment.
    """
    if not text:
        return ""
    
    lines = text.split("\n")
    processed_lines = []
    
    for line in lines:
        # 1. Skip lines that are purely comments or commands that shouldn't be touched
        stripped = line.strip()
        if stripped.startswith("%") or stripped.startswith("\\documentclass"):
            processed_lines.append(line)
            continue
            
        # 2. Escape &, $, #, _ (only if not already escaped)
        for char in ["&", "$", "#", "_"]:
            pattern = f"(?<!\\\\){re.escape(char)}"
            line = re.sub(pattern, "\\" + char, line)
            
        # 3. Escape % ONLY if it's not a comment (preceded by content)
        # and ignore it if it's already part of an escaped sequence.
        if "%" in stripped and not stripped.startswith("%"):
            # Simple check: if % is preceded by something other than a backslash
            line = re.sub(r'(?<!\\)%', r'\%', line)
            
        processed_lines.append(line)
        
    return "\n".join(processed_lines)

def get_llm():
    global _llm
    if _llm is None:
        print("--- Loading NVIDIA NIM Model (405B via OpenAI API) ---")
        _llm = ChatOpenAI(
            model="meta/llama-3.1-405b-instruct",
            openai_api_key=os.getenv("NVIDIA_API_KEY"),
            openai_api_base="https://integrate.api.nvidia.com/v1"
        )
    return _llm

def sanitize_jd_node(state: ResumeState):
    """Phase 2: Remove corporate fluff/boilerplate from the JD."""
    print("--- Node: sanitize_jd ---")
    banned_concepts = [
        "Mission", "Values", "Healthcare", "Insurance", "Synergy", 
        "Culture", "Inclusion", "Benefit", "Diversity", "Equal Opportunity",
        "Comprehensive benefits", "Care-aided", "Pharmacy", "Health outcomes"
    ]
    lines = state['job_description'].split("\n")
    cleaned_lines = []
    for line in lines:
        if not any(word.lower() in line.lower() for word in banned_concepts):
            cleaned_lines.append(line)
    
    cleaned_jd = "\n".join(cleaned_lines)
    return {"job_description": cleaned_jd}

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
        "You are the world's most advanced Technical Resume Editor. Your task is to rewrite a resume to match a specific technical Job Description. "
        "UNBREAKABLE CONTRACT RULES:\n"
        "1. **FACTUAL INTEGRITY**: You MUST NOT change job titles, companies, or dates. If the user is an 'Intern', they MUST remain an 'Intern'.\n"
        "2. **BOILERPLATE PROHIBITION**: Do NOT inject company mission statements, values, or non-technical fluff. Focus ONLY on hard skills and engineering impact.\n"
        "3. **LOGICAL COHERENCE**: Only integrate keywords if they can be tied logically to the existing experience. Do NOT invent new responsibilities.\n"
        "4. **STRUCTURAL PRESERVATION**: Do NOT touch the LaTeX preamble, geometry, or styling. Only modify the content within the sections.\n"
        "5. **SINGLE PAGE**: Ensure the output remains under a single page by being concise and high-impact.\n"
        "6. **LaTeX ESCAPING**: You MUST escape special characters in content text: Use `\\&` for &, `\\%` for %, `\\$` for $, and `\\_` for _. Do NOT escape functional LaTeX characters like `{` or `}`.\n"
        "7. **STRICT OUTPUT**: Return ONLY the raw LaTeX source code. Do NOT include any preamble like 'Here is your code' or 'Updated LaTeX'. Start immediately with `\\documentclass`."
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

    print(f"--- Node: rewrite (Keywords: {len(state['filtered_keywords'])}) ---")
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    print("--- Rewrite Finished ---")
    
    # Clean up response (remove markdown code blocks and conversational fluff)
    new_latex = response.content.strip()
    
    # 1. Strip Markdown code blocks
    if "```latex" in new_latex:
        new_latex = new_latex.split("```latex")[1].split("```")[0]
    elif "```" in new_latex:
        new_latex = new_latex.split("```")[1].split("```")[0]
        
    # 2. Strict Preamble Strip: Find \documentclass and start from there
    if "\\documentclass" in new_latex:
        new_latex = "\\documentclass" + new_latex.split("\\documentclass", 1)[1]
    
    # Post-process to ensure escaping (Safety Net)
    # We only apply this to the text content, avoiding commands. 
    # For now, we apply to the whole string but skip common command patterns.
    new_latex = escape_latex_content(new_latex)
    
    return {"resume_latex": new_latex.strip(), "iteration_count": state.get('iteration_count', 0) + 1}

def validate_content_node(state: ResumeState):
    """Phase 3: Verify seniority and fact preservation."""
    system_prompt = (
        "You are a strict Audit Agent. Compare the 'Original Resume' with the 'Tailored Resume'.\n"
        "Check for the following UNBREAKABLE CONTRACT violations:\n"
        "1. **SENIORITY INFLATION**: Did any job title change from 'Intern', 'Junior', or 'Assistant' to 'Senior', 'Lead', or 'Manager'?\n"
        "2. **FACTUAL HALLUCINATION**: Did the AI add responsibilities or achievements that were not in the original text?\n"
        "3. **TITLE TAMPERING**: Did the AI change the exact wording of job titles or company names?\n\n"
        "If NO violations exist, return ONLY 'PASS'.\n"
        "If violations exist, return 'FAIL: [Description of the lie or inflation]'."
    )
    
    user_prompt = (
        f"Original Resume:\n{state['original_resume_latex']}\n\n"
        f"Tailored Resume:\n{state['resume_latex']}\n\n"
        "Audit Result:"
    )
    
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    result = response.content.strip()
    if "PASS" in result.upper() and len(result) < 10:
        return {"validation_reason": None}
    else:
        return {"validation_reason": result}

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
            print(f"Error details: {error_msg[:1000]}") # Increased for better debugging
            return {"compilation_error": error_msg or "Unknown LaTeX error"}
    except Exception as e:
        print(f"Subprocess exception: {str(e)}")
        return {"compilation_error": str(e)}

def clean_latex(latex_str):
    """
    Strips LaTeX commands to leave only human-readable text.
    This ensures the embedding model compares 'meaning', not 'syntax'.
    """
    if not latex_str:
        return ""
        
    # 1. Remove specific commands but keep content inside (e.g., \textbf{Hello} -> Hello)
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', latex_str)
    
    # 2. Remove standalone commands (e.g., \hfill, \newpage)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # 3. Remove special syntax chars like $, {, }, &, %
    text = re.sub(r'[$&%{}#]', '', text)
    
    # 4. Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def score_node(state: ResumeState):
    """Calculate semantic similarity between JD and tailored resume."""

    clean_resume = clean_latex(state['resume_latex'])
    clean_original = clean_latex(state['original_resume_latex'])
    
    score_model = get_score_model()
    
    # JD Match Score
    emb_jd = score_model.encode(state['job_description'])
    emb_res = score_model.encode(clean_resume)
    jd_score = util.cos_sim(emb_jd, emb_res).item()
    
    # Factual Drift Score (Phase 4)
    # 0.7 is the threshold. If below, we mark as failure to trigger retry.
    emb_orig = score_model.encode(clean_original)
    drift_score = util.cos_sim(emb_orig, emb_res).item()
    
    # If drift is too high (score < 0.7), we store a validation reason for retry
    validation_err = None
    if drift_score < 0.7:
        validation_err = f"Factual Drift Detected ({drift_score:.2f} < 0.7). The meaning has changed too much from the original facts."

    return {
        "semantic_score": jd_score * 100,
        "drift_score": drift_score * 100,
        "validation_reason": validation_err
    }
