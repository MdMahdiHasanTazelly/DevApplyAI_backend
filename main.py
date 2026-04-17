from fastapi import FastAPI
from sentence_transformers import util
from backend.utility import skills_db, extract_skills, model
import re


app = FastAPI()


@app.get("/test")
def testRoute():
    return {"message": "This is a test route"}


@app.post("/analyze")
def analyze(data: dict):
    resume = data["resume"]
    jd = data["jobDesc"]

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
    score = int((similarity * 0.9 + (len(matched)/max(len(jd_skills),1)) * 0.1) * 100)

   # print("Similarity",similarity)

    #  Suggestions [shows top 5 missing skills]
    suggestions = [f"Add {skill} to your resume" for skill in missing[:5]]

    return {
        "score": score,
        "matchedSkills": matched,
        "missingSkills": missing,
        "suggestions": suggestions
    }