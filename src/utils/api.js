import axios from 'axios';

const API_BASE_URL = `${import.meta.env.VITE_API_BASE_URL}/api/v1`;

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor for adding auth token
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
}, (error) => {
    return Promise.reject(error);
});

// Response interceptor for handling errors
api.interceptors.response.use(
    (response) => {
        // If response is successful, return it
        if (response.data && (response.data.data !== undefined || response.data.success !== undefined)) {
            return response.data;
        }
        // If response is not successful, return error
        return {
            success: true,
            data: response.data,
            message: 'Success'
        };
    },
    (error) => {
        // Handle error
        if (error.response?.status === 401) {
            localStorage.removeItem('token');
            localStorage.removeItem('refresh_token');
        }
        return Promise.reject(error);
    }
);

export default api;
