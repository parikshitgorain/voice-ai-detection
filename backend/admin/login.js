// Check if already logged in
const token = localStorage.getItem("adminToken");
if (token) {
  window.location.href = "/admin/";
}

const loginForm = document.getElementById("loginForm");
const errorEl = document.getElementById("error");

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  
  const username = document.getElementById("username").value;
  const password = document.getElementById("password").value;
  
  errorEl.textContent = "";
  
  try {
    const response = await fetch("/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      errorEl.textContent = data.error || "Login failed";
      return;
    }
    
    localStorage.setItem("adminToken", data.token);
    window.location.href = "/admin/";
    
  } catch (err) {
    console.error("Login error:", err);
    errorEl.textContent = "Connection error. Please try again.";
  }
});
