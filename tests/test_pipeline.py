"""Run with:  python -m unittest discover -s tests -v
Requires the model to be trained first (python train_model.py) and reportlab (requirements-dev.txt).
"""
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

from app import app  # noqa: E402
from utils import resume_parser  # noqa: E402
from utils.skill_extractor import extract_technical_skills  # noqa: E402

DS_RESUME = """Alex Sample
alex.sample@example.com | +91 98765 43210 | github.com/alexsample

Summary
Final-year student with hands-on experience in Python, machine learning and data analysis.

Skills
Python, Pandas, NumPy, Scikit-learn, SQL, Matplotlib, Statistics, Git

Projects
Box office predictor: built regression models with Scikit-learn and Pandas, evaluated with cross validation.
Customer churn model: feature engineering, random forest, data visualization with Matplotlib.

Education
MCA, 2026. B.Sc. Computer Science, 2024.

Certifications
Google Data Analytics Certificate
""" * 2


def make_pdf(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in text.splitlines():
        c.drawString(40, y, line[:110]); y -= 14
        if y < 40:
            c.showPage(); y = 800
    c.save()
    return buf.getvalue()


def blank_pdf() -> bytes:
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4); c.rect(50, 50, 200, 200); c.save()
    return buf.getvalue()


class ParserTests(unittest.TestCase):
    def _tmp(self, data: bytes):
        fd, p = tempfile.mkstemp(suffix=".pdf"); os.close(fd)
        with open(p, "wb") as fh:
            fh.write(data)
        self.addCleanup(os.remove, p)
        return p

    def test_extracts_text(self):
        r = resume_parser.extract_text(self._tmp(make_pdf(DS_RESUME)))
        self.assertIn("Scikit-learn", r["text"])

    def test_scanned_pdf_rejected(self):
        with self.assertRaises(resume_parser.ResumeParseError):
            resume_parser.extract_text(self._tmp(blank_pdf()))

    def test_fake_pdf_rejected(self):
        with self.assertRaises(resume_parser.ResumeParseError):
            resume_parser.extract_text(self._tmp(b"not a pdf at all" * 20))

    def test_skill_boundaries(self):
        s = extract_technical_skills("I know JavaScript and C++ but not Java or C.")
        self.assertIn("JavaScript", s); self.assertIn("C++", s); self.assertIn("Java", s)  # 'Java' appears literally
        self.assertNotIn("C#", s)
        self.assertNotIn("Java", extract_technical_skills("Only JavaScript here"))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, data, name="resume.pdf", role="Data Scientist", jd=""):
        form = {"role": role, "job_description": jd}
        if data is not None:
            form["resume"] = (io.BytesIO(data), name)
        return self.client.post("/api/analyze", data=form, content_type="multipart/form-data")

    def test_pages_load(self):
        for path in ["/", "/analyze", "/results", "/model-performance", "/about"]:
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_valid_analysis(self):
        r = self.post(make_pdf(DS_RESUME))
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertTrue(j["ok"])
        self.assertIn("Python", j["skills"]["matched"])
        self.assertTrue(0 <= j["suitability"]["score"] <= 100)
        self.assertTrue(0 <= j["prediction"]["model_probability"] <= 100)
        self.assertEqual(j["prediction"]["predicted_category"] in
                         [x["role"] for x in j["role_comparison"]], True)
        self.assertEqual(sum(j["suitability"]["weights"].values()), 100)

    def test_job_description_changes_target(self):
        jd = "We need someone with SQL, Tableau, Power BI, Excel and Python for dashboards and data cleaning."
        j = self.post(make_pdf(DS_RESUME), role="Other", jd=jd).get_json()
        self.assertTrue(j["target"]["job_description_used"])
        self.assertIn("Tableau", j["skills"]["missing"])
        self.assertIn("SQL", j["skills"]["matched"])

    def test_invalid_inputs(self):
        self.assertEqual(self.post(None).status_code, 400)
        self.assertEqual(self.post(b"hello", name="resume.txt").status_code, 400)
        self.assertEqual(self.post(b"%PDF-1.4 garbage" * 30).status_code, 400)
        self.assertEqual(self.post(blank_pdf()).status_code, 400)
        self.assertEqual(self.post(make_pdf(DS_RESUME), role="Astronaut").status_code, 400)

    def test_too_large(self):
        r = self.post(b"%PDF-" + b"0" * (6 * 1024 * 1024))
        self.assertEqual(r.status_code, 413)

    def test_no_temp_files_left(self):
        before = {f for f in os.listdir(tempfile.gettempdir()) if f.startswith("resume_")}
        self.post(make_pdf(DS_RESUME)); self.post(blank_pdf())
        after = {f for f in os.listdir(tempfile.gettempdir()) if f.startswith("resume_")}
        self.assertEqual(before, after)

    def test_metrics_endpoint_is_real(self):
        j = self.client.get("/api/metrics").get_json()
        self.assertIn("test_metrics", j)


if __name__ == "__main__":
    unittest.main()
