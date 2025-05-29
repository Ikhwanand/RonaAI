import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import PhotoInput from "../components/PhotoInput";
import Waves from "../components/Waves";
import { analysisService } from "../services/analysisService";
import { toast } from "react-toastify";

const AnalysisPage = () => {
  const [image, setImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showPhotoInput, setShowPhotoInput] = useState(true);
  const navigate = useNavigate();

  // Fungsi untuk menangani hasil foto dari PhotoInput
  const handlePhotoTaken = (photoData) => {
    // Jika photoData adalah string (data URL dari kamera)
    if (typeof photoData === 'string') {
      setImage({ dataUrl: photoData });
    } 
    // Jika photoData adalah objek dengan dataUrl (dari ImageUpload)
    else if (photoData && photoData.dataUrl) {
      setImage({ dataUrl: photoData.dataUrl });
    }
    // Jika photoData adalah File (dari ImageUpload)
    else if (photoData instanceof File) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage({ 
          dataUrl: reader.result,
          file: photoData 
        });
      };
      reader.readAsDataURL(photoData);
    }
    
    setShowPhotoInput(false);
  };

  // Fungsi untuk menganalisis gambar
  const handleAnalyzeImage = async () => {
    if (!image || !image.dataUrl) {
      toast.error("Please select or capture an image first");
      return;
    }

    try {
      setIsLoading(true);
      
      // Gunakan file jika tersedia, atau buat blob dari dataUrl
      let imageToAnalyze;
      if (image.file) {
        imageToAnalyze = image.file;
      } else if (image.dataUrl) {
        const blob = await fetch(image.dataUrl).then(r => r.blob());
        imageToAnalyze = new File([blob], 'skin-image.jpg', { type: 'image/jpeg' });
      } else {
        throw new Error("Invalid image format");
      }
      
      const response = await analysisService.analyzeImage(imageToAnalyze);

      if (response.success) {
        const skinType = response.data.analysis.skin_type || 'unknown';
        const slug = typeof skinType === 'string' ? skinType.toLowerCase() : 'unknown';
        navigate(`/results/${response.data.analysis.id}/${slug}`, {
          state: {
            analysisData: response.data.analysis,
            skinProfile: response.data.skin_profile
          }
        });
        toast.success('Analysis completed successfully!');
      }
    } catch (error) {
      toast.error(error.message || "Failed to analyze image");
      console.error("Error analyzing image:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Render komponen
  return (
    <div className="relative min-h-screen">
      <Waves
        lineColor="rgb(132, 127, 245)"
        backgroundColor="rgba(28, 13, 68, 0.2)"
        waveSpeedX={0.02}
        waveSpeedY={0.01}
        waveAmpX={40}
        waveAmpY={20}
        friction={0.9}
        tension={0.01}
        maxCursorMove={120}
        xGap={12}
        yGap={36}
        className="fixed -z-10"
      />
      <motion.div
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <motion.h1
          className="text-3xl font-bold text-center mb-8"
          initial={{ y: -20 }}
          animate={{ y: 0 }}
          transition={{ delay: 0.2 }}
        >
          Skin Analysis
        </motion.h1>

        {showPhotoInput ? (
          <div className="max-w-md mx-auto">
            <PhotoInput 
              onPhotoTaken={handlePhotoTaken} 
              onClose={() => setShowPhotoInput(false)} 
            />
          </div>
        ) : (
          <div className="text-center">
            {image && image.dataUrl ? (
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="mb-6"
              >
                <h2 className="text-xl font-semibold mb-4">Selected Image</h2>
                <div className="bg-white p-4 rounded-lg shadow-md inline-block max-w-md">
                  <img
                    src={image.dataUrl}
                    alt="Selected skin"
                    className="max-h-64 rounded object-contain mx-auto"
                    onError={(e) => {
                      console.error("Image failed to load");
                      e.target.onerror = null;
                      e.target.src = '/placeholder-image.jpg';
                    }}
                  />
                </div>
                
                <div className="mt-6 flex justify-center gap-4">
                  <button
                    className="btn-primary px-8 py-3 text-lg"
                    onClick={handleAnalyzeImage}
                    disabled={isLoading}
                  >
                    {isLoading ? "Analyzing..." : "Analyze Skin"}
                  </button>
                  
                  <button
                    className="btn-secondary px-8 py-3 text-lg"
                    onClick={() => setShowPhotoInput(true)}
                    disabled={isLoading}
                  >
                    Change Photo
                  </button>
                </div>
              </motion.div>
            ) : (
              <div className="text-center">
                <p className="text-red-500 mb-4">No image selected</p>
                <button
                  className="btn-primary px-8 py-3 text-lg"
                  onClick={() => setShowPhotoInput(true)}
                >
                  Select Photo
                </button>
              </div>
            )}
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default AnalysisPage;