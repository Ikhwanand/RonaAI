import api from '../utils/api';

export const journalService = {
    getJournals: async () => {
        try {
            const response = await api.get('/journals/get-journals');
            return response.data;
        } catch (error) {
            console.error('Error fetching journals:', error);
            throw error;
        }
    },

    createJournal: async (journalData) => {
        try {
            const response = await api.post('/journals/create-journal', journalData);
            return {
                success: true,
                data: response.data,
                message: 'Journal created successfully'
            };
        } catch (error) {
            console.error("Error creating journal:", error);
            return {
                success: false,
                message: error.response?.data?.message || error.message || "Failed to create product"
            };
        }
    },

    updateJournal: async (id, journalData) => {
        try {
            const response = await api.put(`/journals/update-journal/${id}`, journalData);
            return {
                success: true,
                data: response.data,
                message: 'Journal updated successfully'
            };
        } catch (error) {
            console.error("Error updating journal:", error);
            return {
                success: false,
                message: error.response?.data?.message || error.message || "Failed to update journal"
            };
        }
    },

    deleteJournal: async (id) => {
        try {
            const response = await api.delete(`/journals/delete-journal/${id}`);
            return {
                success: true,
                data: response,
                message: 'Journal deleted successfully'
            };
        } catch (error) {
            console.error("Error deleting journal:", error);
            return {
                success: false,
                message: error.response?.data?.message || error.message || "Failed to delete journal"
            };
        }
    }
};