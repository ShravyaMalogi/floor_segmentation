// Filter button logic (UI only)
const filters = document.querySelectorAll(".filter");

filters.forEach(btn => {
  btn.addEventListener("click", () => {
    filters.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
  });
});

// Placeholder click handlers
document.querySelectorAll(".texture-card").forEach(card => {
  card.addEventListener("click", () => {
    alert("Texture selected (connect ML logic here)");
  });
});
