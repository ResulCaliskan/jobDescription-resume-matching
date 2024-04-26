from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_resume_score(resume, job_desc):
    # Extract position descriptions from skills and experience
    resume_positions = ' '.join([exp['position'] for exp in resume['experience']])
    job_desc_positions = ' '.join(job_desc.split())

    # Combine position descriptions with other resume content
    resume_text = f"{resume['name']} {resume['title']} {resume_positions}"
    job_desc_text = job_desc

    # Calculate cosine similarity
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform([resume_text, job_desc_text])
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100
    return round(match_percentage, 2)

# Example usage:
resume_data = {
    "name": "Emily Doe",
    "title": "Frontend Developer",
    "contact": {
        "email": "emily.doe@example.com",
        "phone": "+1 418-322-2003",
        "linkedin": "linkedin.com/in/emilydoe"
    },
    "skills": ["JavaScript", "Git", "React", "Django",],
    "experience": [
        {
            "position": "Software Engineer",
            "company": "ABC Tech",
            "duration": "2019 - Present",
            "description": "Developed web applications using Python and Django."
        },
        {
            "position": "Full Stack Developer",
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

job_description = """
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

match_percentage = get_resume_score(resume_data, job_description)
print(f"Match percentage: {match_percentage}%")
