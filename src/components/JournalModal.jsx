import React, { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { toast } from "react-toastify";


const JournalModal = ({ entry, isOpen, onClose, onUpdate }) => {
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (entry) {
      reset({
        title: entry.title,
        content: entry.content,
      });
    } else {
      // Reset form when opening a new entry
      reset({
        title: "",
        content: "",
      });
    }
  }, [entry, reset, isOpen]);

  const onSubmit = async (data) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const journalData = {
        title: data.title,
        content: data.content,
      };

      const response = await onUpdate(journalData);

      if (response && (response.success === true || response.status === 'success')) {
        toast.success(entry ? "Journal entry updated successfully" : "Journal entry added successfully");
        onClose();
      } else {
        setError(response?.message || "Failed to save product");
      }
    } catch (error) {
      console.error("Error in JournalModal:", error);
      setError(error.message || "An unexpected error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white backdrop-blur-md rounded-lg p-6 w-full max-w-md shadow-xl max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold mb-4">
          {entry ? "Edit Journal Entry" : "New Journal Entry"}
        </h2>

        <form onSubmit={handleSubmit(onSubmit)}>
          <div className="mb-4">
            <label htmlFor="journalTitle" className="block text-gray-700 mb-2">Title</label>
            <input
              {...register("title", { required: "Title is required" })}
              id="journalTitle"
              type="text"
              className="w-full p-2 border rounded"
              placeholder="Enter title"
            />
            {errors.title && (
              <p className="text-red-500 text-sm mt-1">
                {errors.title.message}
              </p>
            )}
          </div>

          <div className="mb-4">
            <label htmlFor="journalContent" className="block text-gray-700 mb-2">Content</label>
            <textarea
              {...register("content", { required: "Content is required" })}
              className="w-full p-2 border rounded min-h-[150px]"
              placeholder="Write your journal entry..."
              id="journalContent"
            />
            {errors.content && (
              <p className="text-red-500 text-sm mt-1">
                {errors.content.message}
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
              {isLoading ? "Saving..." : entry ? "Update Entry" : "Save Entry"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default JournalModal;
