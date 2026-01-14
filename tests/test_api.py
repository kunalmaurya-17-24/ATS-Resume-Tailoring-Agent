import requests
import os

def test_tailor_endpoint():
    url = "http://localhost:8000/tailor"
    
    # Path to sample files
    jd_path = "samples/jd.txt"
    resume_path = "samples/resume.tex"
    
    if not os.path.exists(jd_path) or not os.path.exists(resume_path):
        print("Error: Sample files not found. Run from project root.")
        return

    print(f"Testing {url}...")
    
    with open(jd_path, "r") as f:
        jd_text = f.read()
        
    files = {
        "resume_file": open(resume_path, "rb")
    }
    data = {
        "job_description": jd_text
    }
    
    try:
        response = requests.post(url, data=data, files=files)
        response.raise_for_status()
        
        result = response.json()
        print("\n--- Test Results ---")
        print(f"Semantic Score: {result.get('semantic_score'):.2f}%")
        print(f"Keywords Found: {', '.join(result.get('extracted_keywords', []))}")
        print(f"PDF Status: {result.get('pdf_status')}")
        
        if result.get('compilation_error'):
            print(f"Warning: Compilation error captured:\n{result.get('compilation_error')[:200]}...")
        else:
            print("Success: Resume tailored and compiled successfully!")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_tailor_endpoint()
