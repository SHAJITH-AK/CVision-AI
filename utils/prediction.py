"""Loads the trained model once and turns a resume into a full analysis."""
import json
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import resume_parser as rp
from utils.job_profiles import ROLE_PROFILES
from utils.skill_extractor import (extract_soft_skills, extract_technical_skills, find_evidence)
from utils.text_preprocessing import clean_text

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "resume_model.joblib"
METRICS_PATH = Path(__file__).resolve().parent.parent / "models" / "metrics.json"

# ---- documented scoring method (weights sum to 100) ----
WEIGHTS = {
    "required_skills": 45,    # share of required skills evidenced in the resume
    "preferred_skills": 15,   # share of preferred skills evidenced
    "text_similarity": 20,    # TF-IDF cosine similarity to the role / job description
    "experience": 10,         # experience section and detected years
    "education_projects": 10, # education, projects, certifications evidenced
}
SIMILARITY_CAP = 0.5  # cosine >= 0.5 receives full similarity credit (heuristic, not validated)

_lock = threading.Lock()
_model = None


def load_model():
    global _model
    with _lock:
        if _model is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    "Trained model not found. Run `python train_model.py` first (see README).")
            _model = joblib.load(MODEL_PATH)
    return _model


def load_metrics() -> Optional[Dict]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


def _linear_weights(clf) -> Optional[np.ndarray]:
    """Return (n_classes, n_features) weights for LR or a calibrated linear SVM, else None."""
    if hasattr(clf, "coef_"):
        return clf.coef_
    if hasattr(clf, "calibrated_classifiers_"):
        ws = [cc.estimator.coef_ for cc in clf.calibrated_classifiers_ if hasattr(cc.estimator, "coef_")]
        if ws:
            return np.mean(ws, axis=0)
    return None


def _top_terms(model, cleaned: str, class_index: int, n: int = 8) -> List[str]:
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    w = _linear_weights(clf)
    if w is None:
        return []
    x = tfidf.transform([cleaned])
    contrib = x.multiply(w[class_index]).tocoo()
    pairs = sorted(zip(contrib.col, contrib.data), key=lambda t: t[1], reverse=True)
    vocab = tfidf.get_feature_names_out()
    return [vocab[c] for c, v in pairs[:n] if v > 0]


def _coverage(required: List[str], have: set):
    matched = [s for s in required if s in have]
    missing = [s for s in required if s not in have]
    pct = len(matched) / len(required) if required else 0.0
    return matched, missing, pct


def _experience_score(sections: Dict, years: float) -> float:
    s = 0.5 if sections["experience"] else 0.0
    if years >= 1:
        s += 0.25
    if years >= 3:
        s += 0.25
    return s


def _edu_proj_score(sections: Dict) -> float:
    return (0.5 if sections["education"] else 0) + (0.3 if sections["projects"] else 0) + (0.2 if sections["certifications"] else 0)


def _suitability(role_req, role_pref, similarity, sections, years, have):
    _, _, req_pct = _coverage(role_req, have)
    _, _, pref_pct = _coverage(role_pref, have) if role_pref else (None, None, req_pct)
    sim_pct = min(similarity / SIMILARITY_CAP, 1.0)
    exp_pct = _experience_score(sections, years)
    edu_pct = _edu_proj_score(sections)
    parts = {
        "required_skills": req_pct, "preferred_skills": pref_pct, "text_similarity": sim_pct,
        "experience": exp_pct, "education_projects": edu_pct,
    }
    score = sum(WEIGHTS[k] * v for k, v in parts.items())
    return round(score, 1), parts


def analyze(resume_text: str, role: str, job_description: str = "") -> Dict:
    model = load_model()
    tfidf = model.named_steps["tfidf"]
    job_description = (job_description or "").strip()
    cleaned = clean_text(resume_text)

    # --- classification (job category) ---
    proba = model.predict_proba([resume_text])[0]
    classes = list(model.classes_)
    order = np.argsort(proba)[::-1]
    pred_idx = int(order[0])
    predicted = classes[pred_idx]
    top3 = [{"category": classes[i], "probability": round(float(proba[i]) * 100, 1)} for i in order[:3]]
    metrics = load_metrics()
    ece = metrics["test_metrics"].get("expected_calibration_error") if metrics else None

    # --- skills ---
    tech = extract_technical_skills(resume_text)
    soft = extract_soft_skills(resume_text)
    have = set(tech)

    # --- target definition ---
    use_jd = bool(job_description)
    jd_skills = extract_technical_skills(job_description) if use_jd else []
    if role in ROLE_PROFILES and not use_jd:
        prof = ROLE_PROFILES[role]
        required, preferred = prof["required"], prof["preferred"]
        target_text = prof["description"] + " " + " ".join(required + preferred)
        target_label, basis = role, "role profile"
    elif use_jd:
        required = jd_skills   # every skill named in the JD is treated as required
        preferred = []
        target_text = job_description
        target_label = role if role != "Other" else "Custom job description"
        basis = "job description"
    else:  # role == Other without JD -> fall back to the predicted category
        prof = ROLE_PROFILES[predicted]
        required, preferred = prof["required"], prof["preferred"]
        target_text = prof["description"] + " " + " ".join(required + preferred)
        target_label, basis = f"{predicted} (predicted; no role/JD supplied)", "role profile"

    jd_warning = None
    if use_jd and len(jd_skills) < 2:
        jd_warning = ("Fewer than 2 recognisable skills were found in the job description, so the skill-match "
                      "part of the score is unreliable. Paste a longer description or choose a role.")

    sim = float(cosine_similarity(tfidf.transform([cleaned]), tfidf.transform([clean_text(target_text)]))[0, 0])
    sections = rp.detect_sections(resume_text)
    years = rp.estimate_years_experience(resume_text)
    contact = rp.detect_contact(resume_text)

    matched, missing, req_pct = _coverage(required, have)
    pref_matched, pref_missing, _ = _coverage(preferred, have)
    if required:
        score, parts = _suitability(required, preferred, sim, sections, years, have)
    else:  # no skills to compare: skill components are not computable
        score, parts = None, None

    # --- suitability across all supported roles (role profiles only) ---
    role_scores = []
    for r, prof in ROLE_PROFILES.items():
        rs_sim = float(cosine_similarity(tfidf.transform([cleaned]),
                       tfidf.transform([clean_text(prof["description"] + " " + " ".join(prof["required"] + prof["preferred"]))]))[0, 0])
        rs, _ = _suitability(prof["required"], prof["preferred"], rs_sim, sections, years, have)
        role_scores.append({"role": r, "score": rs})
    role_scores.sort(key=lambda d: d["score"], reverse=True)

    # --- evidence & explanation ---
    evidence = [{"skill": s, "snippet": find_evidence(resume_text, s)} for s in matched[:10]]
    top_terms = _top_terms(model, cleaned, pred_idx)
    explanation = [
        f"The classifier assigned the resume to '{predicted}' with a predicted probability of "
        f"{top3[0]['probability']}%." + (f" Top terms that pushed it toward this category: {', '.join(top_terms)}." if top_terms else ""),
        f"{len(matched)} of {len(required)} required skills for the {basis} are evidenced in the resume text.",
    ]
    if role in ROLE_PROFILES and predicted != role and not use_jd:
        explanation.append(f"The model's predicted category ('{predicted}') differs from your selected role ('{role}'). "
                           "This is common when roles share skills; the suitability score above is still computed against your selected role.")
    if score is not None and score < 40:
        explanation.append("The score is low mainly because several required skills are not evidenced in the extracted text.")

    suggestions = []
    for sk in missing[:6]:
        suggestions.append(f"Add or learn {sk}: it is a core requirement for this target and is not evidenced in your resume.")
    if not sections["projects"]:
        suggestions.append("Add a Projects section with 2-3 projects that name the tools you used.")
    if not sections["experience"]:
        suggestions.append("Add internships, freelance work or training experience if you have any, with measurable results.")
    if not sections["certifications"]:
        suggestions.append("List relevant certifications or completed courses if you have them.")
    if not contact["email"]:
        suggestions.append("Make sure a professional email address is clearly visible near the top.")
    if len(resume_text) < 1200:
        suggestions.append("The resume text is quite short; add specifics (tools, outcomes, numbers) to strengthen your evidence.")

    return {
        "prediction": {
            "predicted_category": predicted, "top_predictions": top3,
            "model_probability": top3[0]["probability"], "expected_calibration_error": ece,
            "model_name": metrics["selected_model"] if metrics else type(model).__name__,
        },
        "target": {"role": role, "label": target_label, "basis": basis, "job_description_used": use_jd,
                   "warning": jd_warning},
        "suitability": {
            "score": score, "weights": WEIGHTS, "components": ({k: round(v * 100, 1) for k, v in parts.items()} if parts else None),
            "similarity_cosine": round(sim, 3), "similarity_cap": SIMILARITY_CAP,
            "required_coverage_pct": round(req_pct * 100, 1),
        },
        "skills": {
            "matched": matched, "missing": missing, "preferred_matched": pref_matched, "preferred_missing": pref_missing,
            "technical": tech, "soft": soft, "suggested": (missing + pref_missing)[:8],
        },
        "quality": {
            "contact": contact, "sections": sections, "estimated_years_experience": years,
            "keywords_found": tech[:20],
            "note": "Detection is automatic. 'Not detected' can also mean the PDF layout prevented parsing.",
        },
        "role_comparison": role_scores,
        "evidence": evidence,
        "explanation": explanation,
        "suggestions": suggestions,
        "disclaimer": "Automated estimate for self-assessment only. It is not a hiring decision or a probability of being hired.",
    }
