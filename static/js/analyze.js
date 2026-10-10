(function () {
  const MAX = 5 * 1024 * 1024;
  const $ = id => document.getElementById(id);
  const dz = $('dropzone'), input = $('fileInput'), info = $('fileInfo'), msg = $('msg');
  const btn = $('analyzeBtn'), role = $('role'), jd = $('jd'), prog = $('progress');
  let file = null, extractOk = false;

  function showMsg(text, kind) {
    msg.textContent = text; msg.className = 'alert' + (kind ? ' ' + kind : ''); msg.hidden = !text;
  }
  function refresh() {
    btn.disabled = !(file && extractOk && role.value && (role.value !== 'Other' || jd.value.trim().length > 30));
  }
  function reset() {
    file = null; extractOk = false; input.value = ''; info.hidden = true; showMsg('');
    $('preview').hidden = true; $('previewNote').hidden = true; $('previewEmpty').hidden = false; refresh();
  }
  async function pick(f) {
    reset();
    if (!f) return;
    if (!/\.pdf$/i.test(f.name) && f.type !== 'application/pdf') return showMsg('Only PDF files are accepted.');
    if (f.size > MAX) return showMsg('That file is larger than 5 MB. Upload only your resume.');
    file = f;
    $('fileName').textContent = f.name;
    $('fileMeta').textContent = (f.size / 1024).toFixed(1) + ' KB';
    info.hidden = false;
    showMsg('Reading the PDF…', 'ok');
    const fd = new FormData(); fd.append('resume', f);
    try {
      const r = await fetch('/api/extract', { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok || !d.ok) { file = null; showMsg(d.error || 'Could not read that PDF.'); return; }
      extractOk = true;
      $('fileMeta').textContent = `${(f.size / 1024).toFixed(1)} KB · ${d.pages} page(s) · ${d.characters} characters read`;
      $('preview').textContent = d.preview; $('preview').hidden = false; $('previewEmpty').hidden = true;
      $('previewNote').hidden = !d.truncated; $('previewNote').textContent = 'Preview shows the first 2,500 characters. The full text is analysed.';
      showMsg('');
    } catch (e) {
      showMsg('Could not reach the server. Make sure the Flask app is running.');
    }
    refresh();
  }

  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => pick(input.files[0]));
  ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', e => pick(e.dataTransfer.files[0]));
  $('removeFile').addEventListener('click', reset);
  role.addEventListener('change', refresh); jd.addEventListener('input', refresh);

  btn.addEventListener('click', async () => {
    btn.disabled = true; prog.hidden = false; showMsg('');
    $('progressLabel').textContent = 'Analyzing your resume…';
    const fd = new FormData();
    fd.append('resume', file); fd.append('role', role.value); fd.append('job_description', jd.value);
    try {
      const r = await fetch('/api/analyze', { method: 'POST', body: fd });
      const d = await r.json();
      if (!r.ok || !d.ok) throw new Error(d.error || 'Analysis failed.');
      sessionStorage.setItem('resumeResult', JSON.stringify(d));
      window.location.href = '/results';
    } catch (e) {
      showMsg(e.message || 'Analysis failed.'); prog.hidden = true; refresh();
    }
  });
})();
