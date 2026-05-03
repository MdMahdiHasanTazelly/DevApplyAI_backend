from fastapi import FastAPI, HTTPException
from sentence_transformers import util
from utility import skills_db, extract_skills, model
import requests
from dotenv import load_dotenv
import re

import os

load_dotenv()

TINY_LLAMA_URL = os.getenv("TINY_LLAMA_URL")



app = FastAPI()


@app.get("/test")
def testRoute():
    return {"message": "This is a test route"}


@app.post("/analyze")
def analyze(data: dict):
    resume = data["resume"]
    jd = data["jobDesc"]

    if not resume or not jd:
        raise HTTPException(
            status_code=400,
            error="Resume and Job Description are required."
        )

    #  Overall similarity
    emb1 = model.encode(resume, convert_to_tensor=True)
    emb2 = model.encode(jd, convert_to_tensor=True)

    similarity = float(util.cos_sim(emb1, emb2)[0][0])

    # NEW skill extraction
    resume_skills = extract_skills(resume)
    jd_skills = extract_skills(jd)

    matched = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))

    # Score calculation
    score = int((similarity * 0.2 + (len(matched)/max(len(jd_skills),1)) * 0.8) * 100)

   # print("Similarity",similarity)

    #  Suggestions [shows top 5 missing skills]
    suggestions = [f"Add {skill} to your resume" for skill in missing[:5]]

    return {
        "score": score,
        "matchedSkills": matched,
        "missingSkills": missing,
        "suggestions": suggestions
    }





@app.post("/cv-suggestions")
def generate_cover_letter(data: dict):
    resume = data["resumeText"]
    jobDesc = data["jobDesc"]

    if not resume or not jobDesc:
        raise HTTPException(
            status_code=400,
            error="Resume and Job Description are required."
        )
        

    prompt = f"""
    TASK:
    Write some suggestions for cover letter.
    STRICT RULES:
    - Maximum 100 words
    Resume:
    {resume}

    Job Description:
    {jobDesc}
    """
    
    payload = {
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(TINY_LLAMA_URL, json=payload)
        result = response.json()

        if "response" in result:
            return {
                "cover_letter": result["response"]
            }
        else:
            return {"error": result}
    except Exception as e:
        return {"error": e}




