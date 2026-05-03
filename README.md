# 🚀 DevApplyAI Backend

DevApplyAI is an AI-powered backend service designed to assist developers in automating job application workflows, enhancing resumes, and interacting with AI models for career-related tasks This repository contains the backend implementation built with **FastAPI**, integrating modern Python tools and AI libraries.


## 📁 Project Structure

├── main.py # Entry point of the FastAPI application  
├── utility.py # Helper functions and core logic  
├── README.md # Project documentation  
├── requirements.txt # Python dependencies  
├── pycache/ # Compiled Python files (auto-generated)  
└── venv/ # Virtual environment (should be ignored in Git)  


## ⚙️ Features

- ⚡ FastAPI-based REST API  
- Shows missing skills in the resume from job description.
- Shows similarity score between resume and job description.
- Based on the job desccription and resume, it provide some suggestions to write **cover lettter**.
- 🤖 AI-powered processing (via Transformers / Torch ecosystem)  
- 🌐 Async-ready architecture  


## Models
#### BERT:
For sementic comaprison between job description and resume BERT model is used. Skills are extracted from job description and resume using **Sementic encoding** after that missing skills and matching skills are found from that sementic comparison using BERT models. It also shows the similarity score of job description and resume through cosine similarity.

#### TinyLLAMA: 
For cover letter suggestions tinyllama is used here. From job description and resume it provides guideline to write a cover letter for a particular job.

## API End Points
- `/test` :- To test if the backend is working or not
- `/analyze` :- takes resume & job description as request body and returns the matched, missing skills and similarity score.
- `/cv-suggestions` :- again takes resume & job description as request body and returns cover letter suggestions.

## 🧰 Tech Stack

- **Backend Framework:** FastAPI  
- **Server:** Uvicorn  
- **Language:** Python 3.12   


## 🧠 How It Works  
1. Client sends request to API endpoint.  
2. FastAPI processes the request.  
3. Utility functions handle business logic / AI processing.  
4. Response is returned in JSON format.


## Problems:

1. **Projects virtual environment was not working.**  
**Cause:** Changing the project's folder name. Because in the env configuration file, the directory had remained as the old peoject name. So, global python was running instead of the projects env.  
 **Soln :** Recreated the virtual environment.


## 👨‍💻 Author

**Md Mahdi Hasan Tazelly**  


## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub!


