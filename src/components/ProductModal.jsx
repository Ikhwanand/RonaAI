import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { toast } from "react-toastify";

const ProductModal = ({ product, isOpen, onClose, onUpdate }) => {
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Reset form ketika modal dibuka/ditutup atau product berubah
  useEffect(() => {
    if (isOpen) {
      setError(null); // Clear any previous errors
      reset({
        product_name: product?.product_name || product?.name || "",
        product_category: product?.product_category || product?.category || "",
        ai_recommendation: product?.ai_recommendation || product?.aiRecommendation || false,
      });
    }
  }, [isOpen, product, reset]);

  const onSubmit = async (data) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const productData = {
        product_name: data.product_name,
        product_category: data.product_category,
        ai_recommendation: Boolean(data.ai_recommendation)
      };
      
      // Call the update function and get the response
      const result = await onUpdate(productData);
      
      // Check if the response indicates success
      if (result && (result.success === true || result.status === 'success')) {
        toast.success(product ? "Product updated successfully" : "Product added successfully");
        onClose();
      } else {
        // If not successful, set the error message from the response
        setError(result?.message || "Failed to save product");
      }
    } catch (error) {
      // This will catch any unexpected errors in the onSubmit function itself
      console.error("Error in ProductModal:", error);
      setError(error.message || "An unexpected error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg p-6 w-full max-w-md shadow-xl animate-fade-in">
        <h2 className="text-2xl font-bold mb-4">
          {product ? "Edit Product" : "Add Product"}
        </h2>

        {error && (
          <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-md">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="mb-4">
            <label className="block text-gray-700 mb-2">Product Name</label>
            <input
              {...register("product_name", { required: "Product name is required" })}
              type="text"
              className="w-full p-2 border rounded"
              placeholder="Enter product name"
            />
            {errors.product_name && (
              <p className="text-red-500 text-sm mt-1">
                {errors.product_name.message}
              </p>
            )}
          </div>

          <div className="mb-4">
            <label className="block text-gray-700 mb-2">Category</label>
            <select
              {...register("product_category", { required: "Category is required" })}
              className="w-full p-2 border rounded"
            >
              <option value="">Select a category</option>
              <option value="Cleanser">Cleanser</option>
              <option value="Toner">Toner</option>
              <option value="Serum">Serum</option>
              <option value="Moisturizer">Moisturizer</option>
              <option value="Sunscreen">Sunscreen</option>
              <option value="Mask">Mask</option>
              <option value="Exfoliator">Exfoliator</option>
              <option value="Eye Cream">Eye Cream</option>
              <option value="Treatment">Treatment</option>
              <option value="Other">Other</option>
            </select>
            {errors.product_category && (
              <p className="text-red-500 text-sm mt-1">
                {errors.product_category.message}
              </p>
            )}
          </div>

          <div className="mb-4">
            <label className="flex items-center">
              <input
                {...register("ai_recommendation")}
                type="checkbox"
                className="mr-2"
              />
              <span>AI Recommended</span>
            </label>
          </div>

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-300 rounded hover:bg-gray-400"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50 hover:bg-blue-700"
            >
              {isLoading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Saving...
                </span>
              ) : product ? (
                "Update Product"
              ) : (
                "Add Product"
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ProductModal;