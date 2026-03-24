from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util
import re

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

#  Larger skills database (expand later)
skills_db = [
    # Frontend
    "html", "css", "javascript", "typescript",
    "react", "next.js", "vue", "angular", "svelte",
    "tailwind css", "bootstrap", "material ui",

    # Backend
    "node.js", "express", "django", "flask", "spring boot",
    "fastapi", "nestjs", "laravel"

    # Databases
    "mongodb", "mysql", "postgresql", "sqlite",
    "redis", "firebase", "supabase",

    #  DevOps & Tools
    "docker", "kubernetes", "jenkins", "github actions",
    "git", "nginx", "terraform",

    #  Cloud
    "aws", "azure", "google cloud", "gcp", "digital ocean"

    #  Mobile
    "flutter", "react native", "android", "kotlin", "swift",

    # Programming Languages
    "python", "java", "c", "c++", "c#", "go", "rust", "php", "javascript",

    # AI / ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "keras",

    # Data Science
    "pandas", "numpy", "matplotlib", "seaborn", "data analysis",
    "data visualization", "power bi", "tableau",

    # Security
    "cybersecurity", "penetration testing", "oauth", "jwt",

    #  Testing
    "jest", "mocha", "chai", "cypress", "selenium",

    # Architecture & Concepts
    "rest api", "graphql", "microservices", "system design",

    #  Others
    "linux", "bash", "web scraping", "socket.io"
]

#  Precompute skill embeddings (IMPORTANT)
skill_embeddings = model.encode(skills_db, convert_to_tensor=True)


#  Clean + split text into sentences
def split_sentences(text):
    text = text.lower()
    sentences = re.split(r'[.\n]', text)
    return [s.strip() for s in sentences if s.strip()]


#  Embedding-based skill extraction (sentence by sentence)
def extract_skills(text, threshold=0.5):
    sentences = split_sentences(text)
    found_skills = set()

    for sentence in sentences:
        sentence_embedding = model.encode(sentence, convert_to_tensor=True)

        similarities = util.cos_sim(sentence_embedding, skill_embeddings)[0]

        for i, score in enumerate(similarities):
            if score > threshold:
                found_skills.add(skills_db[i])

    return list(found_skills)

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

    # 🔹 Score calculation
    score = int((similarity * 0.6 + (len(matched)/max(len(jd_skills),1)) * 0.4) * 100)

    # 🔹 Suggestions
    suggestions = [f"Add {skill} to your resume" for skill in missing[:5]]

    return {
        "score": score,
        "matchedSkills": matched,
        "missingSkills": missing,
        "suggestions": suggestions
    }