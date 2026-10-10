"""Skill requirements for each supported role.

These are curated, general-purpose requirement lists (not scraped from any
single employer). They drive the skill-gap analysis and the suitability score
when the user does not paste a specific job description.
"""

ROLE_PROFILES = {
    "Data Scientist": {
        "required": ["Python", "Machine Learning", "Statistics", "Pandas", "NumPy", "Scikit-learn", "SQL", "Data Visualization"],
        "preferred": ["Deep Learning", "TensorFlow", "PyTorch", "Feature Engineering", "NLP", "Model Deployment", "Git"],
        "description": "Build statistical and machine learning models on large datasets, engineer features, validate models, and communicate insights using Python, SQL and visualization tools.",
    },
    "Machine Learning Engineer": {
        "required": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Model Deployment", "Docker", "Git"],
        "preferred": ["Scikit-learn", "Kubernetes", "AWS", "CI/CD", "NLP", "Computer Vision", "REST APIs"],
        "description": "Train, optimise and deploy machine learning and deep learning models to production, building scalable pipelines and serving APIs with Docker and cloud platforms.",
    },
    "Python Developer": {
        "required": ["Python", "OOP", "Git", "SQL", "REST APIs", "Data Structures", "Unit Testing"],
        "preferred": ["Flask", "Django", "FastAPI", "Docker", "Linux", "Databases", "Agile"],
        "description": "Design, develop and test Python applications and backend services using Flask or Django, relational databases, REST APIs and version control.",
    },
    "Web Developer": {
        "required": ["HTML", "CSS", "JavaScript", "Responsive Design", "Git", "REST APIs"],
        "preferred": ["React", "Node.js", "Bootstrap", "Tailwind CSS", "TypeScript", "SQL", "PHP"],
        "description": "Build responsive, accessible websites and web applications with HTML, CSS and JavaScript frameworks, integrating front-end interfaces with back-end APIs.",
    },
    "Data Analyst": {
        "required": ["SQL", "Excel", "Data Visualization", "Statistics", "Data Cleaning", "Python"],
        "preferred": ["Power BI", "Tableau", "Pandas", "Matplotlib", "ETL", "Databases"],
        "description": "Collect, clean and analyse data with SQL, Excel and Python, build dashboards in Power BI or Tableau, and report business insights to stakeholders.",
    },
    "Software Engineer": {
        "required": ["Data Structures", "OOP", "Git", "System Design", "Unit Testing", "Databases"],
        "preferred": ["Java", "Python", "C++", "REST APIs", "Docker", "CI/CD", "Agile", "Linux"],
        "description": "Design and build reliable software systems, apply data structures and algorithms, write tested maintainable code and collaborate in agile teams.",
    },
    "Cybersecurity Analyst": {
        "required": ["Network Security", "Networking", "Vulnerability Assessment", "Incident Response", "Linux", "SIEM"],
        "preferred": ["Penetration Testing", "Wireshark", "Nmap", "Metasploit", "Burp Suite", "OWASP", "Cryptography", "Python"],
        "description": "Monitor networks and systems for threats, assess vulnerabilities, respond to security incidents, and harden infrastructure using SIEM and penetration testing tools.",
    },
    "Cloud Engineer": {
        "required": ["AWS", "Cloud Computing", "Linux", "Docker", "Networking", "CI/CD"],
        "preferred": ["Azure", "GCP", "Kubernetes", "Terraform", "Python", "Bash", "Git"],
        "description": "Provision, automate and operate cloud infrastructure on AWS, Azure or GCP using infrastructure as code, containers and CI/CD pipelines.",
    },
}

# Roles the classifier is trained to predict (same as ROLE_PROFILES keys).
CATEGORIES = list(ROLE_PROFILES.keys())
SELECTABLE_ROLES = CATEGORIES + ["Other"]
