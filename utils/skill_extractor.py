"""Lexicon-based skill extraction.

Only skills that literally appear in the text (as the canonical name or a known
alias) are reported, so every reported skill is backed by evidence.
"""
import re
from typing import Dict, List, Optional

# canonical skill -> aliases (canonical name is always matched too)
TECHNICAL_SKILLS: Dict[str, List[str]] = {
    "Python": ["python3"],
    "Java": [],
    "JavaScript": ["js", "ecmascript", "es6"],
    "TypeScript": [],
    "C++": [],
    "C#": [],
    "Go": ["golang"],
    "Bash": ["shell scripting", "shell script"],
    "PowerShell": [],
    "SQL": ["mysql", "postgresql", "postgres", "sqlite", "sql server", "t-sql"],
    "NoSQL": ["mongodb", "redis", "cassandra", "dynamodb"],
    "R": ["r programming"],
    "HTML": ["html5"],
    "CSS": ["css3", "sass", "scss"],
    "Bootstrap": [],
    "Tailwind CSS": ["tailwind"],
    "React": ["react.js", "reactjs"],
    "Angular": ["angularjs"],
    "Vue": ["vue.js", "vuejs"],
    "Node.js": ["nodejs"],
    "Flask": [],
    "Django": [],
    "FastAPI": [],
    "REST APIs": ["rest api", "restful", "api development"],
    "PHP": [],
    "Pandas": [],
    "NumPy": [],
    "Matplotlib": [],
    "Seaborn": [],
    "Scikit-learn": ["sklearn", "scikit learn"],
    "TensorFlow": [],
    "Keras": [],
    "PyTorch": ["torch"],
    "NLP": ["natural language processing", "spacy", "nltk"],
    "Computer Vision": ["opencv"],
    "Deep Learning": ["neural networks", "cnn", "lstm"],
    "Machine Learning": ["ml", "supervised learning", "classification", "regression"],
    "Statistics": ["statistical analysis", "hypothesis testing", "statistical modeling"],
    "Feature Engineering": [],
    "Model Deployment": ["mlops", "model serving"],
    "Data Visualization": ["data visualisation", "dashboards"],
    "Data Cleaning": ["data wrangling", "data preprocessing", "data preparation"],
    "ETL": ["data pipelines", "data pipeline"],
    "Excel": ["microsoft excel", "advanced excel", "pivot tables"],
    "Power BI": ["powerbi"],
    "Tableau": [],
    "Git": ["github", "gitlab", "version control"],
    "Docker": ["containerization", "containers"],
    "Kubernetes": ["k8s"],
    "Linux": ["ubuntu", "unix"],
    "CI/CD": ["ci/cd", "jenkins", "github actions", "continuous integration"],
    "Terraform": ["infrastructure as code", "iac"],
    "AWS": ["amazon web services", "ec2", "s3", "lambda"],
    "Azure": ["microsoft azure"],
    "GCP": ["google cloud", "google cloud platform"],
    "Cloud Computing": ["cloud infrastructure", "cloud services"],
    "Networking": ["tcp/ip", "dns", "network protocols", "firewalls"],
    "Network Security": ["firewall", "intrusion detection", "vpn"],
    "Penetration Testing": ["pentesting", "pen testing", "ethical hacking", "red team"],
    "Vulnerability Assessment": ["vulnerability scanning", "nessus", "vulnerability management"],
    "Wireshark": [],
    "Nmap": [],
    "Metasploit": [],
    "Burp Suite": [],
    "SIEM": ["splunk", "log analysis", "security monitoring"],
    "Incident Response": ["incident handling"],
    "Cryptography": ["encryption"],
    "OWASP": ["owasp top 10", "web application security"],
    "Data Structures": ["data structures and algorithms", "dsa", "algorithms"],
    "OOP": ["object oriented programming", "object-oriented programming", "object oriented design"],
    "System Design": ["software architecture", "design patterns", "microservices"],
    "Unit Testing": ["pytest", "unittest", "junit", "test driven development", "tdd"],
    "Agile": ["scrum", "kanban"],
    "Databases": ["database design", "database management", "rdbms"],
    "Web Scraping": ["beautifulsoup", "scrapy", "selenium"],
    "Responsive Design": ["responsive web design", "mobile-first"],
    "JSON": [],
    "AJAX": [],
}

SOFT_SKILLS: Dict[str, List[str]] = {
    "Communication": ["communicated", "verbal communication", "written communication"],
    "Teamwork": ["team player", "collaboration", "collaborated", "cross-functional"],
    "Leadership": ["led a team", "team lead", "mentored", "mentoring"],
    "Problem Solving": ["problem-solving", "troubleshooting"],
    "Critical Thinking": ["analytical thinking", "analytical skills"],
    "Time Management": ["deadline", "prioritization", "prioritisation"],
    "Adaptability": ["quick learner", "fast learner", "adaptable"],
    "Presentation": ["presented", "presentations", "public speaking"],
    "Creativity": ["creative thinking", "innovative"],
    "Attention to Detail": ["detail-oriented", "detail oriented"],
}


def _compile(name: str, aliases: List[str]) -> re.Pattern:
    terms = sorted({name.lower(), *[a.lower() for a in aliases]}, key=len, reverse=True)
    alt = "|".join(re.escape(t) for t in terms)
    # Boundaries that treat + # . as part of a token (so "java" != "javascript", "c" != "c++").
    return re.compile(rf"(?<![\w+#.])(?:{alt})(?![\w+#]|\.\w)", re.IGNORECASE)


_TECH_PATTERNS = {n: _compile(n, a) for n, a in TECHNICAL_SKILLS.items()}
_SOFT_PATTERNS = {n: _compile(n, a) for n, a in SOFT_SKILLS.items()}

# Very short ambiguous names need case-sensitive checks to avoid false positives.
_CASE_SENSITIVE = {
    "R": re.compile(r"(?<![\w+#.])R(?![\w+#])(?=\s*(?:,|;|\||/|\)|programming|language|studio|$))"),
    "Go": re.compile(r"(?<![\w+#.])(?:Go|Golang)(?![\w+#])"),
    "Bash": re.compile(r"(?<![\w+#.])[Bb]ash(?![\w+#])"),
}


def _find(patterns: Dict[str, re.Pattern], text: str, skip: Optional[set] = None) -> List[str]:
    found = []
    for name, pat in patterns.items():
        if name in _CASE_SENSITIVE:
            if _CASE_SENSITIVE[name].search(text):
                found.append(name)
            continue
        if pat.search(text):
            found.append(name)
    return found


def extract_technical_skills(text: str) -> List[str]:
    return _find(_TECH_PATTERNS, text or "")


def extract_soft_skills(text: str) -> List[str]:
    return _find(_SOFT_PATTERNS, text or "")


def find_evidence(text: str, skill: str, max_len: int = 150) -> Optional[str]:
    """Return the first line of the resume that mentions the skill."""
    pat = _TECH_PATTERNS.get(skill) or _SOFT_PATTERNS.get(skill)
    if pat is None:
        return None
    for line in (text or "").splitlines():
        line = line.strip()
        if line and pat.search(line):
            return line if len(line) <= max_len else line[: max_len - 1] + "…"
    return None
