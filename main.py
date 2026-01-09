from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

from agent.graph import create_resume_agent

app = FastAPI(title="ATS Resume Tailoring Agent")
agent = create_resume_agent()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/tailor")
async def tailor_resume(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...)
):
    try:
        # Read the LaTeX file content
        resume_content = (await resume_file.read()).decode("utf-8")
        
        # Initial state
        initial_state = {
            "resume_latex": resume_content,
            "original_resume_latex": resume_content,
            "job_description": job_description,
            "extracted_keywords": [],
            "compilation_error": None,
            "semantic_score": 0.0,
            "pdf_path": None,
            "iteration_count": 0
        }
        
        # Run the agent
        final_state = agent.invoke(initial_state)
        
        return {
            "semantic_score": final_state.get("semantic_score"),
            "extracted_keywords": final_state.get("extracted_keywords"),
            "compilation_error": final_state.get("compilation_error"),
            "tailored_latex": final_state.get("resume_latex"),
            "pdf_status": "Success" if final_state.get("pdf_path") else "Failed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download")
def download_pdf():
    pdf_path = "temp_latex/resume.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type='application/pdf', filename="tailored_resume.pdf")
    raise HTTPException(status_code=404, detail="PDF not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
