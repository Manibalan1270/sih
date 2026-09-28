/* ROBOTON theme: night print (default) or day print, the same drafting scale
 * on film or on paper. Runs synchronously in <head>, before the body paints,
 * so there is no flash of the wrong print. */
(() => {
  const KEY = "roboton-theme";
  const root = document.documentElement;

  function apply(theme) {
    if (theme === "light") root.setAttribute("data-theme", "light");
    else root.removeAttribute("data-theme");
  }

  function current() {
    return root.getAttribute("data-theme") === "light" ? "light" : "dark";
  }

  const stored = localStorage.getItem(KEY);
  const initial = stored || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  apply(initial);

  // The button names the print it switches TO, not the one showing now.
  const nextLabel = (theme) => (theme === "light" ? "Night print" : "Day print");

  function wire() {
    const btn = document.getElementById("theme-toggle");
    if (!btn) return;
    btn.textContent = nextLabel(current());
    btn.addEventListener("click", () => {
      const next = current() === "light" ? "dark" : "light";
      apply(next);
      localStorage.setItem(KEY, next);
      btn.textContent = nextLabel(next);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})();
