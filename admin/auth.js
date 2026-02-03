const API_URL = window.location.protocol + '//' + window.location.host;

// Check authentication on protected pages
if (!window.location.pathname.endsWith('login.html')) {
  const token = localStorage.getItem('admin_token');
  
  if (!token) {
    window.location.href = 'login.html';
  } else {
    // Verify token is still valid
    fetch(`${API_URL}/admin/session`, {
      headers: { 'Authorization': `Bearer ${token}` }
    }).then(res => {
      if (!res.ok) {
        localStorage.removeItem('admin_token');
        localStorage.removeItem('admin_username');
        window.location.href = 'login.html';
      }
    }).catch(() => {
      window.location.href = 'login.html';
    });
  }
}

// Logout handler
document.addEventListener('DOMContentLoaded', () => {
  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      localStorage.removeItem('admin_token');
      localStorage.removeItem('admin_username');
      window.location.href = 'login.html';
    });
  }
});

// Helper: Make authenticated API call
async function apiCall(endpoint, options = {}) {
  const token = localStorage.getItem('admin_token');
  
  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json',
    ...options.headers
  };
  
  const res = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers
  });
  
  if (res.status === 401) {
    localStorage.removeItem('admin_token');
    localStorage.removeItem('admin_username');
    window.location.href = 'login.html';
    throw new Error('Unauthorized');
  }
  
  return res;
}
