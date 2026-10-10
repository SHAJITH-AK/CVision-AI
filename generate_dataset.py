"""Generate the SYNTHETIC demonstration dataset (dataset/resumes.csv).

IMPORTANT: every record is machine-generated from templates. None of it is real
candidate data. Because the text comes from templates, the resulting model
metrics are optimistic compared with real-world resumes. Replace
dataset/resumes.csv with real labelled resumes (columns: resume_text, category)
and re-run train_model.py to get realistic numbers.

Usage:  python generate_dataset.py
"""
import json
import random
from pathlib import Path

import pandas as pd

from utils.job_profiles import ROLE_PROFILES

SEED = 42
PER_CLASS = 160
OUT = Path(__file__).parent / "dataset" / "resumes.csv"

# Role-specific work bullets. {skill} placeholders are filled from the role's skills.
BULLETS = {
    "Data Scientist": [
        "Built predictive models using {s} to forecast customer churn and improved recall by 12 percent",
        "Performed exploratory data analysis and feature engineering on datasets with over 1 million rows",
        "Designed A/B tests and applied hypothesis testing to evaluate product changes",
        "Trained and tuned gradient boosting and random forest models with cross validation",
        "Presented model findings and statistical insights to business stakeholders",
        "Created visualizations with {s} to explain model behaviour",
    ],
    "Machine Learning Engineer": [
        "Deployed deep learning models as REST APIs served in {s} containers",
        "Optimised neural network training pipelines and reduced training time by 30 percent",
        "Built model serving and monitoring workflows for production machine learning systems",
        "Implemented computer vision and NLP models using {s}",
        "Automated retraining pipelines with CI/CD and experiment tracking",
        "Converted research prototypes into scalable inference services",
    ],
    "Python Developer": [
        "Developed backend services in Python using {s} and relational databases",
        "Wrote unit tests with pytest and maintained over 85 percent code coverage",
        "Designed REST APIs and integrated third party services",
        "Refactored legacy scripts into object oriented modules",
        "Automated internal workflows with Python scripts and scheduled jobs",
        "Optimised SQL queries and database access layers",
    ],
    "Web Developer": [
        "Built responsive websites using {s} that work across desktop and mobile",
        "Developed interactive user interfaces with JavaScript and component based frameworks",
        "Integrated front end pages with back end REST APIs",
        "Improved page load time and accessibility scores across client sites",
        "Implemented cross browser layouts with flexbox and grid",
        "Collaborated with designers to convert mockups into working pages",
    ],
    "Data Analyst": [
        "Wrote SQL queries to extract and join data from multiple sources for weekly reporting",
        "Built interactive dashboards in {s} used by the sales and operations teams",
        "Cleaned and validated raw data, removing duplicates and handling missing values",
        "Analysed trends and reported key performance indicators to management",
        "Automated recurring Excel reports and reduced manual effort by 10 hours per week",
        "Applied descriptive statistics to identify business opportunities",
    ],
    "Software Engineer": [
        "Designed and implemented modular software components applying data structures and algorithms",
        "Participated in design reviews and system design discussions for scalable services",
        "Wrote maintainable object oriented code in {s} with automated tests",
        "Worked in an agile team with code reviews, sprint planning and CI pipelines",
        "Debugged production issues and improved reliability of core services",
        "Contributed to database schema design and API contracts",
    ],
    "Cybersecurity Analyst": [
        "Monitored network traffic and security alerts using {s} to detect suspicious activity",
        "Performed vulnerability assessments and reported remediation steps to system owners",
        "Responded to security incidents following documented incident response procedures",
        "Conducted penetration testing of web applications following OWASP guidelines",
        "Hardened Linux servers and reviewed firewall rules",
        "Analysed logs in a SIEM and wrote detection rules",
    ],
    "Cloud Engineer": [
        "Provisioned and managed cloud infrastructure on {s} using infrastructure as code",
        "Built CI/CD pipelines to automate build, test and deployment of containerised services",
        "Configured virtual networks, load balancers, and security groups",
        "Monitored cloud resources and reduced monthly spend by 18 percent",
        "Containerised applications with Docker and orchestrated them on Kubernetes",
        "Automated routine operations with Bash and Python scripts",
    ],
}

# Roles that realistically share vocabulary; used to inject overlap/noise.
NEIGHBOURS = {
    "Data Scientist": ["Machine Learning Engineer", "Data Analyst", "Python Developer"],
    "Machine Learning Engineer": ["Data Scientist", "Software Engineer", "Cloud Engineer"],
    "Python Developer": ["Software Engineer", "Web Developer", "Data Scientist"],
    "Web Developer": ["Software Engineer", "Python Developer"],
    "Data Analyst": ["Data Scientist", "Python Developer"],
    "Software Engineer": ["Python Developer", "Web Developer", "Cloud Engineer"],
    "Cybersecurity Analyst": ["Cloud Engineer", "Software Engineer"],
    "Cloud Engineer": ["Cybersecurity Analyst", "Software Engineer", "Machine Learning Engineer"],
}

DEGREES = ["B.Sc. Computer Science", "B.Tech Information Technology", "MCA", "B.E. Computer Engineering",
           "M.Sc. Data Science", "B.C.A.", "B.Sc. Information Systems", "M.Sc. Computer Science"]
GENERIC_TITLES = ["Software Developer", "Technical Associate", "Engineer", "IT Analyst", "Associate Engineer",
                  "Technology Trainee", "Developer", "Systems Associate"]
GENERIC_BULLETS = [
    "Collaborated with team members to deliver features on schedule",
    "Documented processes and presented progress updates in weekly meetings",
    "Investigated issues, fixed defects and improved existing solutions",
    "Worked with stakeholders to gather requirements and validate results",
    "Used Git for version control and participated in code reviews",
    "Learned new tools quickly and supported colleagues with technical questions",
]
CERTS = ["AWS Cloud Practitioner", "Google Data Analytics Certificate", "CompTIA Security+", "Azure Fundamentals",
         "Coursera Machine Learning Specialization", "Oracle Java SE Certification", "freeCodeCamp Responsive Web Design",
         "Certified Ethical Hacker (CEH)", "Python Institute PCAP", "Microsoft Power BI Data Analyst"]
SOFT = ["strong communication and teamwork", "problem solving and attention to detail", "quick learner who adapts fast",
        "analytical thinking and time management", "collaborates in cross-functional teams"]
GENERIC_PROJECTS = ["Library management system", "Student attendance tracker", "Inventory dashboard",
                    "Expense tracker application", "Quiz portal", "Weather information app"]
COMPANIES = ["Northwind Labs", "BlueOrbit Systems", "Kestrel Analytics", "Lumen Digital", "Vertex Soft",
             "Pinecone Works", "Orchid Technologies", "Meridian IT Services"]


def make_resume(role: str, rng: random.Random) -> str:
    prof = ROLE_PROFILES[role]
    core = prof["required"] + prof["preferred"]
    k = rng.randint(max(3, len(core) // 3), max(4, int(len(core) * 0.8)))
    skills = set(rng.sample(core, min(k, len(core))))
    # overlap: borrow skills from neighbouring roles
    bullets = rng.sample(BULLETS[role], rng.randint(1, 4))
    bullets += rng.sample(GENERIC_BULLETS, rng.randint(1, 3))
    if rng.random() < 0.8:
        other = rng.choice(NEIGHBOURS[role])
        oc = ROLE_PROFILES[other]["required"] + ROLE_PROFILES[other]["preferred"]
        skills.update(rng.sample(oc, rng.randint(2, 5)))
        bullets += rng.sample(BULLETS[other], rng.randint(1, 3))
    if rng.random() < 0.3:  # unrelated skills from any role
        anyrole = rng.choice(list(ROLE_PROFILES))
        ac = ROLE_PROFILES[anyrole]["required"] + ROLE_PROFILES[anyrole]["preferred"]
        skills.update(rng.sample(ac, 3))
    rng.shuffle(bullets)
    skills.update(rng.sample(["Git", "Linux", "SQL", "Python", "Excel", "Agile"], rng.randint(0, 3)))
    skills = sorted(skills)
    rng.shuffle(skills)
    sample_skill = lambda: rng.choice(skills)  # noqa: E731
    bullets = [b.replace("{s}", sample_skill()) for b in bullets]

    parts = []
    parts.append(f"Summary\nMotivated candidate seeking a technical position with hands on experience in "
                 f"{', '.join(skills[:3])}.")
    parts.append("Skills\n" + ", ".join(skills))
    if rng.random() < 0.8:  # some candidates have no experience section (fresher resumes)
        title = rng.choice(GENERIC_TITLES)
        yrs = rng.choice(["2021 - 2023", "2022 - 2024", "2020 - 2022", "2023 - 2025", "2019 - 2023"])
        parts.append(f"Work Experience\n{title}, {rng.choice(COMPANIES)} ({yrs})\n- " + "\n- ".join(bullets))
    if rng.random() < 0.75:
        proj = rng.sample(GENERIC_PROJECTS, 1)[0]
        parts.append(f"Projects\n{proj}: built using {sample_skill()} and {sample_skill()}.")
    parts.append(f"Education\n{rng.choice(DEGREES)}, {rng.choice(['2019', '2021', '2023', '2024'])}")
    if rng.random() < 0.5:
        parts.append("Certifications\n" + rng.choice(CERTS))
    if rng.random() < 0.6:
        parts.append("Strengths\n" + rng.choice(SOFT))
    return "\n\n".join(parts)


def main():
    rng = random.Random(SEED)
    rows = []
    for role in ROLE_PROFILES:
        for _ in range(PER_CLASS):
            rows.append({"resume_text": make_resume(role, rng), "category": role})
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    info = {"type": "synthetic", "records": len(df), "seed": SEED,
            "note": "Template-generated demonstration data. Not real candidates."}
    (OUT.parent / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"Wrote {len(df)} SYNTHETIC records to {OUT}")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
