import os
import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from utils import prediction, resume_parser
from utils.job_profiles import SELECTABLE_ROLES

MAX_UPLOAD_MB = 5
MAX_JD_CHARS = 10000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False


# ---------------- helpers ----------------
def _wants_json() -> bool:
    return request.path.startswith("/api/")


def _error(message: str, status: int):
    if _wants_json():
        return jsonify({"ok": False, "error": message}), status
    return render_template("error.html", message=message, status=status), status


def _read_upload():
    """Validate the uploaded file, parse it, and always delete the temp copy."""
    f = request.files.get("resume")
    if f is None or not f.filename:
        raise resume_parser.ResumeParseError("No file received. Choose a PDF resume to upload.")
    if not f.filename.lower().endswith(".pdf"):
        raise resume_parser.ResumeParseError("Only PDF files are accepted. Export your resume as a PDF.")
    fd, tmp_path = tempfile.mkstemp(prefix=f"resume_{uuid.uuid4().hex}_", suffix=".pdf")
    os.close(fd)  # Windows: release the handle before other libraries open the file
    try:
        f.save(tmp_path)
        size_kb = round(os.path.getsize(tmp_path) / 1024, 1)
        parsed = resume_parser.extract_text(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    parsed["filename"] = Path(f.filename).name[:120]
    parsed["size_kb"] = size_kb
    return parsed


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------- pages ----------------
@app.route("/")
def index():
    return render_template("index.html", active="home")


@app.route("/analyze")
def analyze_page():
    return render_template("analyze.html", active="analyze", roles=SELECTABLE_ROLES, max_mb=MAX_UPLOAD_MB)


@app.route("/results")
def results_page():
    # Results are rendered in the browser from the /api/analyze response; nothing is stored on the server.
    return render_template("results.html", active="results")


@app.route("/model-performance")
def model_performance():
    return render_template("model_performance.html", active="model", m=prediction.load_metrics())


@app.route("/about")
def about():
    return render_template("about.html", active="about", metrics=prediction.load_metrics(),
                           weights=prediction.WEIGHTS, cap=prediction.SIMILARITY_CAP)


# ---------------- API ----------------
@app.route("/api/roles")
def api_roles():
    return jsonify({"roles": SELECTABLE_ROLES})


@app.route("/api/extract", methods=["POST"])
def api_extract():
    try:
        parsed = _read_upload()
    except resume_parser.ResumeParseError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    text = parsed["text"]
    return jsonify({"ok": True, "filename": parsed["filename"], "pages": parsed["pages"],
                    "characters": parsed["characters"], "size_kb": parsed["size_kb"], "preview": text[:2500],
                    "truncated": len(text) > 2500})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    role = request.form.get("role", "").strip()
    jd = request.form.get("job_description", "").strip()
    if role not in SELECTABLE_ROLES:
        return jsonify({"ok": False, "error": "Select a job role from the list."}), 400
    if len(jd) > MAX_JD_CHARS:
        return jsonify({"ok": False, "error": f"The job description is too long (limit {MAX_JD_CHARS} characters)."}), 400
    try:
        parsed = _read_upload()
        result = prediction.analyze(parsed["text"], role, jd)
    except resume_parser.ResumeParseError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    result["file"] = {"name": parsed["filename"], "pages": parsed["pages"], "characters": parsed["characters"]}
    result["ok"] = True
    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    m = prediction.load_metrics()
    if not m:
        return jsonify({"ok": False, "error": "No metrics found. Run train_model.py."}), 404
    return jsonify(m)


# ---------------- errors ----------------
@app.errorhandler(413)
def too_large(_):
    return _error(f"The file is larger than the {MAX_UPLOAD_MB} MB limit.", 413)


@app.errorhandler(404)
def not_found(_):
    return _error("That page does not exist.", 404)


@app.errorhandler(405)
def bad_method(_):
    return _error("That action is not allowed here.", 405)


@app.errorhandler(500)
def server_error(_):
    return _error("Something went wrong on the server while processing your request.", 500)


if __name__ == "__main__":
    # Fail early with a clear message if the model has not been trained yet.
    try:
        prediction.load_model()
    except FileNotFoundError as exc:
        print(f"\n[!] {exc}\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
