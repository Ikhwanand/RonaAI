import React, { useRef, useState } from "react";

const ImageUpload = ({ onUpload, onClose }) => {
  const fileInputRef = useRef(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && validateFile(file)) {
      processFile(file);
    }
  };

  const validateFile = (file) => {
    const validTypes = ['image/jpeg', 'image/png', 'image/gif'];
    if (!validTypes.includes(file.type)) {
      alert('Please upload a valid image (JPEG, PNG, GIF)');
      return false;
    }
    if (file.size > 5 * 1024 * 1024) {
      alert('File size should be less than 5MB');
      return false;
    }
    return true;
  };

  const processFile = (file) => {
    const reader = new FileReader();
    reader.onload = () => {
      setPreviewImage(reader.result);
      onUpload(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="card p-4">
      <h2 className="text-xl font-semibold mb-4">Upload an Image</h2>
      
      <div className="space-y-4">
        <div
          className={`border-2 border-dashed rounded-lg p-6 text-center ${
            dragActive
              ? "border-primary bg-primary bg-opacity-10"
              : "border-gray-300"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {previewImage ? (
            <div className="space-y-4">
              <img
                src={previewImage}
                alt="Preview"
                className="max-h-64 mx-auto rounded-lg"
              />
              <button
                className="btn-secondary"
                onClick={() => {
                  setPreviewImage(null);
                  if (fileInputRef.current) {
                    fileInputRef.current.value = "";
                  }
                }}
              >
                Remove Image
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="text-lightText">
                <p>Drag and drop your image here, or</p>
              </div>
              <button
                className="btn-primary"
                onClick={() => fileInputRef.current.click()}
              >
                Browse Files
              </button>
            </div>
          )}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            className="hidden"
          />
        </div>
        <button 
          className="btn-secondary w-full mt-4" 
          onClick={onClose}
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

export default ImageUpload;
