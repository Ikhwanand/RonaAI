import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";

const DeleteAccountModal = ({ isOpen, onClose, onDelete }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const navigate = useNavigate();

  const handleDelete = async () => {
    if (confirmText !== "DELETE") {
      toast.error("Please type DELETE to confirm");
      return;
    }

    try {
      setIsLoading(true);
      const success = await onDelete(); // Get the result from onDelete
      
      if (success) {
        // Only show success message and redirect if deletion was successful
        localStorage.removeItem("token");
        localStorage.removeItem("refresh_token"); // Make sure this matches what's used in authService
        toast.success("Account deleted successfully");
        navigate("/login");
      }
      // If not successful, the error toast will be shown by the parent component
    } catch (error) {
      // Only show error if it wasn't already shown by the parent
      if (!error.handled) {
        toast.error("Failed to delete account");
      }
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white backdrop-blur-md rounded-lg p-6 w-full max-w-md shadow-xl">
        <h2 className="text-2xl font-bold mb-4 text-red-600">Delete Account</h2>
        <p className="text-gray-600 mb-6">
          This action cannot be undone. All your data, including skin analysis
          history, journal entries, and saved products will be permanently
          deleted.
        </p>

        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Type DELETE to confirm
          </label>
          <input
            type="text"
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            className="w-full p-2 border rounded focus:ring-red-500 focus:border-red-500"
            placeholder="Type DELETE"
          />
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
            onClick={handleDelete}
            disabled={isLoading || confirmText !== "DELETE"}
            className="px-4 py-2 bg-red-600 text-white rounded disabled:opacity-50 hover:bg-red-700"
          >
            {isLoading ? "Deleting..." : "Delete Account"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DeleteAccountModal;
