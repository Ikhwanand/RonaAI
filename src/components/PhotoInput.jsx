import React, { useState } from "react";
import Camera from "./Camera";
import ImageUpload from "./ImageUpload";

const PhotoInput = ({ onPhotoTaken, onClose }) => {
  const [mode, setMode] = useState(null); // 'camera' or 'upload'

  const handlePhotoCaptured = (photoData) => {
    onPhotoTaken(photoData);
    onClose();
  };

  if (mode === 'camera') {
    return <Camera onCapture={handlePhotoCaptured} onClose={() => setMode(null)} />;
  }

  if (mode === 'upload') {
    return <ImageUpload onUpload={handlePhotoCaptured} onClose={() => setMode(null)} />;
  }

  return (
    <div className="card p-4">
      <h2 className="text-xl font-semibold mb-4">Add Photo</h2>
      <div className="space-y-4">
        <button 
          className="btn-primary w-full"
          onClick={() => setMode('camera')}
        >
          Take Photo
        </button>
        <button 
          className="btn-primary w-full"
          onClick={() => setMode('upload')}
        >
          Upload Photo
        </button>
        <button 
          className="btn-secondary w-full"
          onClick={onClose}
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

export default PhotoInput;