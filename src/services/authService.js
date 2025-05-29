import api from "../utils/api";

export const authService = {
  register: async (userData) => {
    const response = await api.post("/auth/register", {
      name: userData.name,
      email: userData.email,
      country: userData.country,
      password: userData.password,
    });
    return response;
  },

  login: async (email, password) => {
    const params = new URLSearchParams();
    params.append('username', email);
    params.append('password', password);
  
    try {
      const response = await api.post('/auth/login', params, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });
      
      const token = response?.access_token || response?.data?.access_token;
      const refreshToken = response?.refresh_token || response?.data?.refresh_token;
      if (response?.data?.access_token) {
        localStorage.setItem("token", token);
        localStorage.setItem('refresh_token', refreshToken); 
      }
  
      if (response?.access_token) {
        throw new Error('Invalid server response format');
      }
      return response;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  },

  refreshToken: async (refreshToken) => {
    const response = await api.post("/auth/refresh", {
      refresh_token: refreshToken,
    });
    if (response) {
      localStorage.setItem("token", response.access_token);
      localStorage.setItem("refresh_token", response.refresh_token);
    }
    return response;
  },

  logout: () => {
    localStorage.removeItem("token");
    localStorage.removeItem("refresh_token");
  },
};
