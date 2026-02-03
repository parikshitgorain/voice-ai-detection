const API_URL = window.location.protocol + '//' + window.location.host;

document.getElementById('loginForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;
  const errorEl = document.getElementById('error');
  
  errorEl.textContent = '';
  
  try {
    const res = await fetch(`${API_URL}/admin/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    
    const data = await res.json();
    
    if (!res.ok) {
      errorEl.textContent = data.error || 'Login failed';
      return;
    }
    
    localStorage.setItem('admin_token', data.token);
    localStorage.setItem('admin_username', data.username);
    window.location.href = 'dashboard.html';
    
  } catch (err) {
    errorEl.textContent = 'Connection error. Please try again.';
  }
});
