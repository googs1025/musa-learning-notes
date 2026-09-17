document.addEventListener("DOMContentLoaded", () => {
  const search = document.querySelector("[data-knowledge-search]");
  const cards = [...document.querySelectorAll("[data-search-text]")];
  const empty = document.querySelector("[data-empty-state]");
  if (!search || cards.length === 0) return;
  const applyFilter = () => {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    for (const card of cards) {
      const matched = card.dataset.searchText.toLowerCase().includes(query);
      card.classList.toggle("is-hidden", !matched);
      if (matched) visible += 1;
    }
    if (empty) empty.hidden = visible !== 0;
  };
  search.addEventListener("input", applyFilter);
});
