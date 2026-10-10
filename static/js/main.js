// Shared helpers + mobile navigation
(function () {
  const btn = document.getElementById('navToggle');
  const bar = document.getElementById('sidebar');
  if (btn && bar) {
    btn.addEventListener('click', () => {
      const open = bar.classList.toggle('open');
      btn.setAttribute('aria-expanded', String(open));
    });
  }
})();

// Safe DOM helpers (all dynamic text goes through textContent).
window.fillList = function (id, items) {
  const ul = document.getElementById(id);
  ul.replaceChildren(...(items || []).map(t => { const li = document.createElement('li'); li.textContent = t; return li; }));
};
