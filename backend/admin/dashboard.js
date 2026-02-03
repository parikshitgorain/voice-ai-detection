// Dashboard logic
const token = localStorage.getItem("adminToken");
if (!token) {
  window.location.href = "/admin/login.html";
}

const logoutBtn = document.getElementById("logoutBtn");

// Load stats
async function loadStats() {
  try {
    const response = await fetch("/admin/stats", {
      headers: { "Authorization": `Bearer ${token}` }
    });
    
    if (!response.ok) {
      if (response.status === 401) {
        localStorage.removeItem("adminToken");
        window.location.href = "/admin/login.html";
        return;
      }
      throw new Error("Failed to load stats");
    }
    
    const data = await response.json();
    
    // Update stats
    document.getElementById("totalKeys").textContent = data.stats.total_keys || 0;
    document.getElementById("activeKeys").textContent = data.stats.active_keys || 0;
    document.getElementById("totalRequests").textContent = data.stats.total_requests || 0;
    document.getElementById("todayRequests").textContent = data.stats.today_requests || 0;
    
    // Render activity table
    renderActivity(data.keys);
  } catch (err) {
    console.error("Error loading stats:", err);
  }
}

// Render activity table
function renderActivity(keys) {
  const activityTableBody = document.getElementById("activityTableBody");
  
  // Sort keys by last_used (usage is inside each key object now)
  const sortedKeys = keys.sort((a, b) => {
    const aTime = a.usage && a.usage.last_used ? new Date(a.usage.last_used).getTime() : 0;
    const bTime = b.usage && b.usage.last_used ? new Date(b.usage.last_used).getTime() : 0;
    return bTime - aTime;
  }).slice(0, 10); // Show top 10
  
  if (sortedKeys.length === 0) {
    activityTableBody.innerHTML = '<tr><td colspan="4" style="text-align: center;">No activity yet</td></tr>';
    return;
  }
  
  activityTableBody.innerHTML = sortedKeys.map(key => {
    const usage = key.usage || {};
    const lastUsed = usage.last_used 
      ? new Date(usage.last_used).toLocaleString() 
      : "Never";
    
    return `
      <tr>
        <td>${key.name || "Unnamed"}</td>
        <td>${usage.total_requests || 0}</td>
        <td>${usage.today_requests || 0}</td>
        <td>${lastUsed}</td>
      </tr>
    `;
  }).join("");
}

// Logout
logoutBtn.addEventListener("click", (e) => {
  e.preventDefault();
  localStorage.removeItem("adminToken");
  window.location.href = "/admin/login.html";
});

// Load stats on page load
loadStats();

// Refresh stats every 30 seconds
setInterval(loadStats, 30000);
