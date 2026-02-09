// Filter button logic (UI only)
const filters = document.querySelectorAll(".filter");

filters.forEach(btn => {
  btn.addEventListener("click", () => {
    filters.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
  });
});

// Set up repeating texture backgrounds
document.addEventListener("DOMContentLoaded", () => {
  const textureCards = document.querySelectorAll(".texture-card");
  
  textureCards.forEach(card => {
    const img = card.querySelector("img");
    const preview = card.querySelector(".texture-card-preview");
    
    if (img && preview) {
      // Set the background image from the img src
      preview.style.backgroundImage = `url('${img.src}')`;
    }
  });
});

// Placeholder click handlers
document.querySelectorAll(".texture-card").forEach(card => {
  card.addEventListener("click", () => {
    alert("Texture selected (connect ML logic here)");
  });
});