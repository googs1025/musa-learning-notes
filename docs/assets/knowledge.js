document.addEventListener("DOMContentLoaded", () => {
  const search = document.querySelector("[data-knowledge-search]");
  const cards = [...document.querySelectorAll("[data-search-text]")];
  const empty = document.querySelector("[data-empty-state]");
  const status = document.querySelector("[data-search-status]");
  const controls = document.querySelector("[data-search-controls]");
  if (!search || cards.length === 0) return;
  const entries = cards.map((card) => ({
    card,
    searchText: `${card.textContent} ${card.dataset.searchText}`.toLowerCase(),
  }));
  const applyFilter = () => {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    for (const { card, searchText } of entries) {
      const matched = searchText.includes(query);
      card.classList.toggle("is-hidden", !matched);
      if (matched) visible += 1;
    }
    if (empty) empty.hidden = visible !== 0;
    if (status) status.textContent = visible === 0 ? "没有匹配内容。" : `找到 ${visible} 条内容。`;
  };
  search.addEventListener("input", applyFilter);
  if (controls) controls.hidden = false;
  applyFilter();
});
