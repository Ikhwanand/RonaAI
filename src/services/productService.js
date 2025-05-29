import api from '../utils/api';

export const productService = {
    getProducts: async () => {
        try {
            const response = await api.get('/products/get-products');
            return response.data;
        } catch (error) {
            console.error("Error fetching products:", error);
            throw error;
        }
    },

    createProduct: async (productData) => {
        try {
            const response = await api.post('/products/create-product', productData);
            return {
                success: true,
                data: response.data,
                message: 'Product created successfully'
            };
        } catch (error) {
            console.error("Error creating product:", error);
            return {
                success: false,
                message: error.response?.data?.message || error.message || "Failed to create product"
            };
        }
    },

    updateProduct: async (id, productData) => {
        try {
            const response = await api.put(`/products/update-product/${id}`, productData);
            return {
                success: true,
                data: response.data,
                message: 'Product updated successfully'
            };
        } catch (error) {
            console.error("Error updating product:", error);
            return {
                success: false,
                message: error.response?.data?.message || error.message || "Failed to update product"
            };
        }
    },

    deleteProduct: async (id) => {
        try {
            const response = await api.delete(`/products/delete-product/${id}`);
            return {
                success: true,
                data: response,
                message: 'Product deleted successfully'
            };
        } catch (error) {
            console.error("Error deleting product:", error);
            return {
                success: false,
                message: error.response?.data?.message || error.message || "Failed to delete product"
            };
        }
    }
};