import api from '../utils/api';

export const skinService = {
    getSkinProfile: async () => {
        try {
            const response = await api.get('/skin/get-profile-skin');
            return response;
        } catch (error) {
            console.error('Error fetching skin profile:', error);
            throw error;
        }
    },

    createSkinProfile: async (skinData) => {
        try {
            const concernsArray = typeof skinData.concerns === 'string' 
            ? skinData.concerns.split(',').map((item => item.trim()))
            : skinData.concerns;

            const response = await api.post('/skin/create-profile-skin', {
                skin_type: skinData.skin_type,
                concerns: concernsArray
            });
            return {
                success: true,
                data: response.data
            };
        } catch (error) {
            console.error('Error creating skin profile:', error);
            throw error;
        }
    },

    updateSkinProfile: async (skinData) => {
        try {
            const response = await api.put(`/skin/update-skin-profile`, {
                skin_type: skinData.skin_type,
                concerns: skinData.concerns
            });
            return {
                success: true,
                data: response.data,
                message: response.data?.message || 'Skin profile updated successfully'
            };
        } catch (error) {
            console.error('Error updating skin profile:', error);
            throw error;            
        }
    },

};