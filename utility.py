import re

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

#  Larger skills database 
skills_db = [
    # Frontend
    "html", "css", "javascript", "typescript",
    "react", "next.js", "vue", "angular", "svelte",
    "tailwind css", "bootstrap", "material ui",

    # Backend
    "node.js", "express", "django", "flask", "spring boot",
    "fastapi", "nestjs", "laravel", "golang", "ruby on rails"

    # Databases
    "mongodb", "mysql", "postgresql", "sqlite", "maria db", "elasticsearch"
    "redis", "firebase", "supabase", "cassandra",

    #  DevOps & Tools
    "docker", "kubernetes", "jenkins", "github actions", "kafka", "rabbit mq",
    "git", "nginx", "terraform", "grafana", 

    #  Cloud
    "aws", "azure", "google cloud", "gcp", "digital ocean"

    #  Mobile
    "flutter", "react native", "android", "kotlin", "swift",

    # Programming Languages
    "python", "java", "c", "c++", "c#", "go", "rust", "php", "javascript", "go",

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
    "rest api", "graphql", "microservices", "system design", "soap"

    #  Others
    "linux", "bash", "web scraping", "socket.io"
]



#  Precompute skill embeddings 
skill_embeddings = model.encode(skills_db, convert_to_tensor=True)


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


#  Clean + split text into sentences
def split_sentences(text):
    text = text.lower()
    sentences = re.split(r'[.\n]', text)
    return [s.strip() for s in sentences if s.strip()]

