import React, { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { toast } from "react-toastify";

const SkinModal = ({ skin, isOpen, onClose, onUpdate }) => {
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm();
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (skin) {
      reset({
        skin_type: skin.skin_type,
        concerns: skin.concerns,
      });
    }
  }, [skin, reset]);

  const onSubmit = async (data) => {
    try {
      setIsLoading(true);
      const formattedData = {
        skin_type: data.skin_type,
        concerns: Array.isArray(data.concerns) ? data.concerns : [data.concerns]
      };

      const response = await onUpdate(formattedData);

      if (response?.success) {
        toast.success(response.message || "Skin profile updated successfully");
        onClose();
      } else {
        throw new Error(response?.message || "Failed to update skin profile");
      }
    } catch (error) {
      console.error("Update error:", error);
      toast.error(error.response?.data?.detail || error.response?.data?.message || error.message || "Failed to update skin profile");
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  const skinTypes = ["Oily", "Dry", "Combination", "Normal"];

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white backdrop-blur-md rounded-lg p-6 w-full max-w-md shadow-xl max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">Update Skin Profile</h2>

        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="mb-4">
            <label className="block text-gray-700 mb-2">Skin Type</label>
            <select
              {...register("skin_type", {
                required: "Skin type is required",
              })}
              className="w-full p-2 border rounded bg-white"
            >
              <option value="">Select skin type</option>
              {skinTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
            {errors.skin_type && (
              <p className="text-red-500 text-sm mt-1">
                {errors.skin_type.message}
              </p>
            )}
          </div>

          <div className="mb-4">
            <label className="block text-gray-700 mb-2">Skin Concerns</label>
            <textarea
              {...register("concerns", {
                required: "Skin concerns are required",
              })}
              className="w-full p-2 border rounded min-h-[100px"
              placeholder="Enter your skin concerns (e.g., Acne, Dryness, Dark spots)"
            />
            {errors.concerns && (
              <p className="text-red-500 text-sm mt-1">
                {errors.concerns.message}
              </p>
            )}
          </div>

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-300 rounded hover:bg-gray-400"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50 hover:bg-blue-700"
            >
              {isLoading ? "Updating..." : "Update"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SkinModal;
