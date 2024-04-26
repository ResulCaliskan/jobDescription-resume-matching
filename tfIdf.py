
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Load the Spacy model for text processing
nlp = spacy.load('en_core_web_sm')

# Sample job description
job_description = """Remote Part-time Frontend Developer (1-3 Years Experience)
About the Company
[Company Name] is a fast growing software company.We meet the needs of our customers by providing user-oriented, innovative solutions.
Job Description
As a Remote Part-time Frontend Developer, you will join our team and have some of the following responsibilities:

Designing and developing user-friendly and engaging web interfaces
Create interactive and responsive web applications using JavaScript, React and other related web technologies
Apply UI/UX best practices to maximize user experience
Client-side optimizations and performance improvements
Collaborate with other team members and ensure seamless integration
Technical Skills

JavaScript, React
Django (optional)
HTML, CSS, Sass/Less
TypeScript knowledge is a plus
Solid understanding of web browsers and DOM
Knowledge of responsive design principles
Git version control system
Experience with Agile methodologies
Good communication and interpersonal skills
Education and Experience

Bachelor's degree in Computer Science, Software Development or a related field or equivalent experience
3-5 years of proven experience in frontend development
A solid foundation in JavaScript and React
Working Hours

This is a part-time position and approximately 20-25 hours of work per week is requested. Flexibility on working hours will be provided.

Application
To apply for this exciting opportunity, please send your resume and additional information to [email address]."""

# CVs
cvs = [
{
    "name": "Emily Doe",
    "title": "Frontend Developer",
    "contact": {
        "email": "emily.doe@example.com",
        "phone": "+1 418-322-2003",
        "linkedin": "linkedin.com/in/emilydoe"
    },
    "skills": ["Vue", "JavaScript", "React", "Django", "Next.js"],
    "experience": [
        {
            "position": "Frontend Developer",
            "company": "ABC Tech",
            "duration": "2019 - Present",
            "description": "Developed web applications using React and Next.js."
        },
        {
            "position": "Frontend Developer",
            "company": "Tech Innovators",
            "duration": "2017 - 2019",
            "description": "Worked on React-based frontend development."
        }
    ],
    "education": [
        {
            "degree": "Master of Computer Science",
            "university": "Data Science School",
            "graduation_year": 2015
        }
    ]
}   
]

# Preprocess job description
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

job_description_preprocessed = preprocess_text(job_description)
print("Nlp responzii"+job_description_preprocessed)
# Preprocess CVs
resumes_preprocessed = []
for cv in cvs:
    skills = ' '.join(cv["skills"])
    experience = ' '.join([exp["position"] for exp in cv["experience"]])
    education = ' '.join([edu["degree"] for edu in cv["education"]])
    description = ' '.join([exp["description"] for exp in cv["experience"]])
    # contact = ' '.join([cv["contact"]["email"], cv["contact"]["phone"], cv["contact"]["linkedin"]])
    full_text = ' '.join([cv["title"], skills, experience, description,education])
    resumes_preprocessed.append(preprocess_text(full_text))


cv = preprocess_text(resumes_preprocessed[0])
print("\nResume proceed:"+cv)
# Modify the TF-IDF vectorizer to increase the weight of matching skills


vectorizer = TfidfVectorizer(tokenizer=str.split)  # Use whitespace tokenizer to preserve skill phrases
tfidf_matrix = vectorizer.fit_transform([job_description_preprocessed] + [cv])


# Get the feature names (terms) from the vectorizer
feature_names = np.array(vectorizer.get_feature_names_out())

# Find indices of skill terms in the feature names
skill_indices = [np.where(feature_names == skill.lower())[0][0] for skill in cvs[0]["skills"] if skill.lower() in feature_names]

# Increase the IDF weights for skill terms
for idx in skill_indices:
    # Increase the IDF weight for skill terms by multiplying with a factor
    idf_weight = tfidf_matrix.getcol(idx).toarray().mean()  # Get the average IDF weight for the skill term
    tfidf_matrix[:, idx] *= (idf_weight * 30)  # Adjust the IDF weight by multiplying with a factor (e.g., 2)

# Apply LSI (Truncated SVD) to reduce dimensionality
n_components = 10  # Increase the number of components for better representation
lsa = TruncatedSVD(n_components=n_components)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# Calculate cosine similarity between job description and resumes
job_vector = lsa_matrix[0].reshape(1, -1)
resume_vectors = lsa_matrix[1:]

similarities = cosine_similarity(job_vector, resume_vectors)
similarity_scores = similarities.flatten()

# Sort resumes by similarity score (higher score means better match)
sorted_resumes = [resume for _, resume in sorted(zip(similarity_scores, cvs), reverse=True)]

# Additional complexity: Create a DataFrame to store results
df_results = pd.DataFrame({
    "Resume": [resume["name"] for resume in sorted_resumes],
    "Similarity Score": similarity_scores,
})

# Additional complexity: Add a random noise factor to similarity scores
noise_factor = np.random.uniform(0.9, 1.1, len(similarity_scores))
df_results["Adjusted Score"] = df_results["Similarity Score"]

# Print the enhanced results
print("Top Matching Resumes:")
print(df_results[["Resume", "Adjusted Score"]])

# Additional complexity: Visualize the results (requires matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.barh(df_results["Resume"], df_results["Adjusted Score"], color="skyblue")
plt.xlabel("Adjusted Similarity Score")
plt.ylabel("Resumes")
plt.title("Job Resume Matching Results")
plt.gca().invert_yaxis()
plt.show()
