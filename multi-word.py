from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

# Load the pre-trained vectors
model_path = "./GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Define weighting factors for skills and experience positions
skill_weight = 0.8
experience_weight = 0.2

# Resume and job description data
resume_data = {
    "name": "Emily Doe",
    "title": "Frontend Developer",
    "contact": {
        "email": "emily.doe@example.com",
        "phone": "+1 418-322-2003",
        "linkedin": "linkedin.com/in/emilydoe"
    },
    "skills": ["Vue", "JavaScript", "CSS", "HTML", "React", "Django", "Next.js"],
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

job_desc = """
İş Tanımı: Frontend Geliştirici (5-10 Yıl Tecrübe)

Amaç:

5-10 yıllık deneyime sahip, dinamik ve yenilikçi bir Frontend Geliştirici arıyoruz. Başvuru sahipleri, modern web teknolojilerinde, özellikle React, Vue, Next.js ve kullanıcı deneyimi tasarımı (UX) alanlarında uzman olmalıdır.

Sorumluluklar:

React ve Vue.js gibi frontend çerçevelerini kullanarak kullanıcı arayüzleri (UI) geliştirmek
Düzenli, bakım yapılabilir ve yeniden kullanılabilir kod yazmak
Kullanıcı deneyimi (UX) ilkelerini uygulamak
Next.js'i kullanarak statik ve sunucu taraflı web uygulamaları oluşturmak
Diğer geliştiriciler ve tasarımcılarla işbirliği yapmak
Testler yazmak ve kod kalitesini sağlamak
En son web teknolojilerini takip etmek ve bunları projelerde uygulamak
Teknik Beceriler:

React, Vue.js, Next.js ve Node.js'te derinlemesine bilgi
HTML, CSS ve JavaScript'te uzmanlık
Kullanıcı deneyimi (UX) ilkeleri bilgisi
Versiyon kontrol sistemleri (ör. Git) bilgisi
Tasarım araçları (ör. Figma, Adobe XD) ile deneyim
Performans optimizasyonu ve erişilebilirlik konularında bilgi
Kişisel Nitelikler:

Mükemmel iletişim becerileri
Güçlü analitik ve problem çözme becerileri
Ayrıntılara dikkat etme ve mükemmellik için çabalama
Ekip çalışmasına yatkınlık
Hızlı öğrenme ve yeni teknolojileri benimseme yeteneği
İş Şartları:

Tam zamanlı pozisyon
Rekabetçi maaş ve yan haklar
Esnek çalışma saatleri
Modern ve dinamik bir çalışma ortamı
Sürekli öğrenme ve gelişim fırsatları"""

# Tokenize and preprocess the text
resume_tokens = [token.lower() for token in resume_data["skills"]]
resume_tokens.extend([token.lower() for exp in resume_data["experience"] for token in exp["position"].split()])
job_desc_tokens = [token.lower() for token in job_desc.split()]

# Calculate weighted average for skills
skills_vector = [model[token] * skill_weight for token in resume_tokens if token in model]
skills_vector_avg = sum(skills_vector) / len(skills_vector) if skills_vector else None

# Calculate weighted average for experience positions
experience_vector = [model[token] * experience_weight for token in resume_tokens if token in model]
experience_vector_avg = sum(experience_vector) / len(experience_vector) if experience_vector else None

# Tokenize and preprocess the job description text
job_desc_tokens = [token.lower() for token in job_desc.split()]

# Calculate word embeddings for job description
job_desc_vectors = [model[token] for token in job_desc_tokens if token in model]

# Combine vectors if both are available
if skills_vector_avg is not None and experience_vector_avg is not None:
    combined_vector = (skills_vector_avg + experience_vector_avg) / 2
elif skills_vector_avg is not None:
    combined_vector = skills_vector_avg
elif experience_vector_avg is not None:
    combined_vector = experience_vector_avg
else:
    combined_vector = None

# Compute cosine similarity
if combined_vector is not None:
    similarity_matrix = cosine_similarity([combined_vector], job_desc_vectors)
    average_similarity = similarity_matrix.mean()
    match_percentage = average_similarity * 100
    print("Match percentage:", match_percentage)
else:
    print("No relevant skills or experience found in the resume.")
