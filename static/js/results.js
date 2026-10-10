(function () {
  const $ = id => document.getElementById(id);
  let d = null;
  try { d = JSON.parse(sessionStorage.getItem('resumeResult') || 'null'); } catch (e) { d = null; }
  if (!d) { $('empty').hidden = false; return; }
  $('report').hidden = false;

  const P = d.prediction, S = d.suitability, K = d.skills, Q = d.quality, T = d.target;
  $('meta').textContent = `${d.file.name} · ${d.file.pages} page(s) · ${d.file.characters} characters analysed`;
  if (T.warning) { $('jdWarn').textContent = T.warning; $('jdWarn').hidden = false; }

  // summary
  if (S.score === null) {
    $('scoreNum').textContent = '–'; $('scoreSub').textContent = 'Not computed: no skills could be read from the target description.';
  } else {
    $('scoreRing').style.setProperty('--p', S.score); $('scoreNum').textContent = Math.round(S.score);
    $('scoreRing').setAttribute('aria-label', `Suitability score ${S.score} out of 100`);
    $('scoreSub').textContent = 'Out of 100. A heuristic estimate, not a hiring decision.';
  }
  $('predCat').textContent = P.predicted_category;
  $('predTop').textContent = 'Next most likely: ' + P.top_predictions.slice(1).map(t => `${t.category} (${t.probability}%)`).join(', ');
  $('predProb').textContent = P.model_probability + '%';
  $('probNote').textContent = P.expected_calibration_error != null
    ? `Probability of the predicted category. Calibration error on test data: ${P.expected_calibration_error}.` : 'Probability of the predicted category.';
  $('targetRole').textContent = T.label;
  $('targetBasis').textContent = 'Compared against the ' + T.basis + '.';
  fillList('explain', d.explanation);

  // skills
  fillList('matched', K.matched); fillList('missing', K.missing);
  fillList('prefMatched', K.preferred_matched); fillList('prefMissing', K.preferred_missing);
  fillList('tech', K.technical); fillList('soft', K.soft); fillList('suggested', K.suggested);

  // quality
  const rows = [
    ['Email address', Q.contact.email], ['Phone number', Q.contact.phone], ['LinkedIn / GitHub link', Q.contact.profile_link],
    ['Education', Q.sections.education], ['Work experience', Q.sections.experience],
    ['Projects', Q.sections.projects], ['Certifications', Q.sections.certifications], ['Skills section', Q.sections.skills],
  ];
  $('checks').replaceChildren(...rows.map(([label, ok]) => {
    const li = document.createElement('li');
    const st = document.createElement('span'); st.className = 'st ' + (ok ? 'yes' : 'no'); st.textContent = ok ? 'Detected' : 'Not detected';
    const tx = document.createElement('span'); tx.textContent = label; li.append(st, tx); return li;
  }));
  $('qualNote').textContent = `${Q.note} Estimated experience: ${Q.estimated_years_experience || 0} year(s) (best effort).`;

  $('evidence').replaceChildren(...(d.evidence.length ? d.evidence : [{ skill: 'No matched skills to show', snippet: '' }]).map(e => {
    const li = document.createElement('li'); const s = document.createElement('strong'); s.textContent = e.skill; li.append(s);
    if (e.snippet) { const q = document.createElement('q'); q.textContent = e.snippet; li.append(q); } return li;
  }));
  fillList('suggestions', d.suggestions.length ? d.suggestions : ['No major gaps detected against this target.']);

  // method text
  const W = S.weights, C = S.components;
  const ORDER = ['required_skills', 'preferred_skills', 'text_similarity', 'experience', 'education_projects'];
  const names = { required_skills: 'Required skills', preferred_skills: 'Preferred skills', text_similarity: 'Text similarity', experience: 'Experience', education_projects: 'Education, projects, certifications' };
  const ul = document.createElement('ul'); ul.className = 'plain';
  ORDER.forEach(k => { const li = document.createElement('li'); li.textContent = `${names[k]}: weight ${W[k]}` + (C ? `, your component score ${C[k]}/100` : ''); ul.append(li); });
  $('method').replaceChildren(ul);
  $('disclaimer').textContent = `TF-IDF cosine similarity: ${S.similarity_cosine} (capped at ${S.similarity_cap} for full credit). ` + d.disclaimer;

  // charts
  const PINK = '#EC4899', LAV = '#D9A8E8', SLATE = '#1F2937';
  Chart.defaults.font.family = '"Segoe UI",system-ui,sans-serif'; Chart.defaults.color = SLATE;
  new Chart($('chDonut'), { type: 'doughnut',
    data: { labels: ['Matched', 'Missing'], datasets: [{ data: [K.matched.length, K.missing.length], backgroundColor: [PINK, LAV], borderWidth: 0 }] },
    options: { maintainAspectRatio: false, cutout: '65%', plugins: { legend: { position: 'bottom' }, tooltip: { enabled: true } } } });
  if (C) new Chart($('chComp'), { type: 'bar',
    data: { labels: ['Required', 'Preferred', 'Similarity', 'Experience', 'Education'], datasets: [{ data: ORDER.map(k => C[k]), backgroundColor: PINK, borderRadius: 6 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 100 } } } });
  new Chart($('chRoles'), { type: 'bar',
    data: { labels: d.role_comparison.map(r => r.role), datasets: [{ data: d.role_comparison.map(r => r.score), backgroundColor: d.role_comparison.map(r => r.role === T.role ? PINK : LAV), borderRadius: 6 }] },
    options: { indexAxis: 'y', maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 100 } } } });
})();
