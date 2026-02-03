async function loadStats() {
  try {
    const res = await apiCall('/admin/stats');
    const data = await res.json();
    
    document.getElementById('totalKeys').textContent = data.total_keys;
    document.getElementById('activeKeys').textContent = data.active_keys;
    document.getElementById('totalCalls').textContent = data.total_calls.toLocaleString();
    document.getElementById('todayCalls').textContent = data.today_calls.toLocaleString();
  } catch (err) {
    console.error('Failed to load stats:', err);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  setInterval(loadStats, 30000); // Refresh every 30 seconds
});
