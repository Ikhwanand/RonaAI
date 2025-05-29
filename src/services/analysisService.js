import api from '../utils/api';

export const analysisService = {
    analyzeImage: async (imageFile) => {
       const formData = new FormData();

       // Handle different input types consistently
       if (imageFile instanceof File) {
         formData.append('image', imageFile);
       } else if (typeof imageFile === 'string' && imageFile.startsWith('data:')) {
         const blob = await fetch(imageFile).then(r => r.blob());
         formData.append('image', blob, 'skin-image.png');
       } else if (imageFile instanceof FormData) {
         // If FormData is already provided, use it directly
         return analyzeWithFormData(imageFile);
       } else {
         throw new Error('Invalid image format');
       }

       return analyzeWithFormData(formData);
    },

    getAnalysis: async (analysisId) => {
        try {
            const response = await api.get(`/analysis/get-analysis/${analysisId}`);
            return {
                success: true,
                data: response.data,
                message: response.data.message || 'Analysis retrieved successfully'
            };
        } catch (error) {
            console.error('Error getting analysis:', error);
            throw {
                success: false,
                message: error.response?.data?.message || 'Failed to get analysis',
                error: error.response?.data?.errors || error.message
            };
        }
    },

    getProgress: async (analysisId) => {
        const response = await api.get(`/progress/metrics/${analysisId}`);
        return response;
    },

    getAnalysisHistory: async (skip = 0, limit = 10) => {
        try {
            const response = await api.get('/analysis/history', {
                params: { skip, limit }
            });
            
            if (response.data?.items) {
                response.data.items = response.data.items.map(item => ({
                    ...item,
                    imageUrl: `${import.meta.env.VITE_API_URL || ''}${item.image_url}`
                }));
            }
            // Pastikan kita mengembalikan response.data, bukan response
            return response.data;
        } catch (error) {
            console.error('Error fetching analysis history:', error);
            throw error;
        }
    },
    
    deleteAnalysis: async (id) => {
        const response = await api.delete(`/analysis/delete-analysis/${id}`);
        return response;
    }

};

// Helper function to avoid code duplication
async function analyzeWithFormData(formData) {
  try {
    const response = await api.post('/analysis/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return {
      success: true,
      data: response.data,
      message: response.data.message || 'Analysis completed successfully'
    };
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw {
      success: false,
      message: error.response?.data?.message || 'Failed to analyze image',
      error: error.response?.data?.errors || error.message
    };
  }
}