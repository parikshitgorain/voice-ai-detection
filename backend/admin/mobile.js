// Mobile menu toggle
function toggleMobileMenu() {
  const nav = document.getElementById('adminNav');
  if (nav) {
    nav.classList.toggle('mobile-open');
  }
}

// Close mobile menu when clicking outside
document.addEventListener('click', function(event) {
  const nav = document.getElementById('adminNav');
  const menuBtn = document.querySelector('.mobile-menu-btn');
  
  if (nav && menuBtn && nav.classList.contains('mobile-open')) {
    if (!nav.contains(event.target) && !menuBtn.contains(event.target)) {
      nav.classList.remove('mobile-open');
    }
  }
});

// Close mobile menu when window is resized to desktop
window.addEventListener('resize', function() {
  if (window.innerWidth > 768) {
    const nav = document.getElementById('adminNav');
    if (nav) {
      nav.classList.remove('mobile-open');
    }
  }
});
