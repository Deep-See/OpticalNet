document.addEventListener('DOMContentLoaded', () => {
    const container = document.querySelector('.image-comparison-content');
    const slider = document.querySelector('.slider');
    const image2 = document.querySelector('.image-2');
    const caption = document.querySelector('.caption-text');
  
    if (!container || !slider || !image2 || !caption) {
      console.error('Required elements for image comparison are missing.');
      return;
    }
  
    container.addEventListener('mousemove', (e) => {
      const rect = container.getBoundingClientRect();
      let xPos = e.clientX - rect.left;
      xPos = Math.max(0, Math.min(xPos, rect.width));
      const percentage = (xPos / rect.width) * 100;
  
      slider.style.left = `${percentage}%`;
      image2.style.clipPath = `inset(0 0 0 ${percentage}%)`;

      // Update caption based on slider position
      if (percentage < 50) {
        caption.textContent = "Real object";
      } else {
        caption.textContent = "🔬Observed at subwavelength";
      }
    });
});