from typing import TypedDict, List, Optional

class ResumeState(TypedDict):
    resume_latex: str      # The current LaTeX code
    original_resume_latex: str # The original LaTeX code before any modifications
    job_description: str   # The target JD
    extracted_keywords: list[str] # Changed from List[str]
    filtered_keywords: list[str] # Keywords after filtering
    compilation_error: str # Store error logs if compile fails (changed from Optional[str])
    validation_reason: str # Reason for validation failure or success
    semantic_score: float  # Final alignment score
    drift_score: float     # Score indicating how much the resume drifted from original
    pdf_path: str          # Path to generated PDF (changed from Optional[str])
    iteration_count: int   # Track fix attempts
