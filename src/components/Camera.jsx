import React, { useRef, useState, useCallback, useEffect } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import { toast } from "react-toastify";

const Camera = ({ onCapture, onClose }) => {
  const webcamRef = useRef(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [model, setModel] = useState(null);
  const [isFaceDetected, setIsFaceDetected] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  // const [capturedImage, setCapturedImage] = useState(null);
  const detectionIntervalRef = useRef(null);

  // Load the face detection model
  useEffect(() => {
    const loadModel = async () => {
      setIsModelLoading(true);
      try {
        const loadedModel = await tf.loadGraphModel('/model-tfjs/model.json');
        console.log('Model loaded successfully');
        setModel(loadedModel);
      } catch (error) {
        console.error('Failed to load model:', error);
      } finally {
        setIsModelLoading(false);
      }
    };

    loadModel();

    // Clean up the model
    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
    };
  }, []);

  // Start face detection when camera becomes active
  useEffect(() => {
    if (isCameraActive && model && !isModelLoading) {
      startDetection();
    }
    
    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
    };
  }, [isCameraActive, model, isModelLoading]);

  // Start to detect face
  const startDetection = () => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }

    // Start to detect the face with interval 100ms 
    detectionIntervalRef.current = setInterval(() => {
      detectFace();
    }, 100);
  };

  // Detect face real time
  // const detectFace = async () => {
  //   if (!model || !webcamRef.current || !webcamRef.current.video.readyState === 4) return;

  //   try {
  //     // Take picture from webcam and convert to tensor
  //     const video = webcamRef.current.video;
  //     const videoWidth = video.videoWidth;
  //     const videoHeight = video.videoHeight;

  //     // Set the dimensions of the video
  //     webcamRef.current.video.width = videoWidth;
  //     webcamRef.current.video.height = videoHeight;

  //     // Create a tensor from webcam image
  //     const img = tf.browser.fromPixels(video);

  //     // Process the image 
  //     let processedImg = img;

  //     // Padding the image if need it
  //     if (img.shape[0] > img.shape[1]) {
  //       processedImg = tf.pad(img, [[0, 0], [0, img.shape[0] - img.shape[1]], [0, 0]], 0);
  //     } else {
  //       processedImg = tf.pad(img, [[0, img.shape[1] - img.shape[0]], [0, 0], [0, 0]], 0);
  //     }

  //     // Resize the image
  //     const resizedImg = tf.image.resizeBilinear(processedImg, [512, 512]);

  //     // Add batch dimension
  //     const batchedImg = resizedImg.expandDims(0);

  //     // Run the model
  //     const predictions = await model.execute(batchedImg);

  //     // Process prediction result
  //     const faces = await processPredictions(predictions);

  //     // Update state based on detection result
  //     setIsFaceDetected(faces.length > 0);

  //     // Clear up the tensor for memory management
  //     img.dispose();
  //     processedImg.dispose();
  //     resizedImg.dispose();
  //     batchedImg.dispose();

  //   } catch (error) {
  //     console.error('Error detect face:', error);
  //   }
  // };

  const detectFace = async () => {
    // Check if model exists and webcam is ready
    if (!model || !webcamRef.current) return;

    const video = webcamRef.current.video;

    // Add comprehensive video readness check
    if (!video || !video.readyState || video.readyState !== 4 ||
      video.videoWidth === 0 || video.videoHeight === 0
    ) {
      return;
    }

    try {
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      // Set video dimensions
      video.width = videoWidth;
      video.height = videoHeight;

      // Create tensor only when we have valid dimensions
      const img = tf.browser.fromPixels(video);

      // Validate tensor dimensions
      if (img.shape[0] === 0 || img.shape[1] === 0) {
        img.dispose();
        return;
      }

      // Process the image
      let processedImg = img;
      
      // Padding the image if needed
      if (img.shape[0] > img.shape[1]) {
        processedImg = tf.pad(img, [[0, 0], [0, img.shape[0] - img.shape[1]], [0, 0]], 0);
      } else {
        processedImg = tf.pad(img, [[0, img.shape[1] - img.shape[0]], [0, 0], [0, 0]], 0);
      }

      // Resize the image
      const resizedImg = tf.image.resizeBilinear(processedImg, [512, 512]);
      const batchedImg = resizedImg.expandDims(0);

      // Run the model
      const predictions = await model.execute(batchedImg);
      const faces = await processPredictions(predictions);

      // Update state based on detection result
      setIsFaceDetected(faces.length > 0);

      // Cleanup tensors
      tf.dispose([img, processedImg, resizedImg, batchedImg]);

    } catch (err) {
      console.error('Error detect face:', err);
    }
  }

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user"
  };

  const capture = useCallback(() => {
    if (!isFaceDetected) {
      toast.error('Face not detected! Make sure you are looking at the camera.')
      return;
    }

    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      onCapture(imageSrc);
    }
  }, [webcamRef, onCapture, isFaceDetected]);

  return (
    <div className="card p-4">
      <h2 className="text-xl font-semibold mb-4">Take a Photo</h2>

      {!isCameraActive ? (
        <div className="space-y-4">
          <button 
            className="btn-primary w-full" 
            onClick={() => setIsCameraActive(true)}
          >
            Activate Camera
          </button>
          <button className="btn-secondary w-full" onClick={onClose}>
            Cancel
          </button>
        </div>
      ) : (
        isModelLoading ? (
          <div className="flex justify-center items-center h-48">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-400 mx-auto mb-2"></div>
              <p className="text-gray-600">Loading model...</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              className="w-full rounded-lg"
            />
            <div className="flex items-center justify-center mb-2">
              <span className={`inline-block w-3 h-3 rounded-full mr-2 ${isFaceDetected ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <span className="text-sm">{isFaceDetected ? 'Face detected' : 'Face not detected'}</span>
            </div>
            <div className="flex gap-2">
              <button 
                className={`btn-primary flex-1 ${!isFaceDetected ? 'opacity-50' : ''}`} 
                onClick={capture}
                disabled={!isFaceDetected}
              >
                Capture Photo
              </button>
              <button 
                className="btn-secondary flex-1" 
                onClick={() => setIsCameraActive(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        )
      )}
    </div>
  );
};


 // Process predictions from the model
 const processPredictions = async (predictions) => {
  try {
    // Get the detection results
    let boxes, scores;
    
    if (Array.isArray(predictions)) {
      // Format output model 
      if (predictions.length >= 2) {
        if (predictions[0].shape.length === 4) {
          boxes = predictions[0];
          scores = predictions[1];
        } else {
          boxes = predictions[1];
          scores = predictions[0];
        }
      } else {
        // If model just have 1 output
        boxes = predictions[0];
        scores = tf.ones([boxes.shape[0], boxes.shape[1]]);
      }
    } else {
      // If predictions is not an array
      boxes = predictions;
      scores = tf.ones([boxes.shape[0], boxes.shape[1]]);
    }

    // Convert tensor to array
    const boxesArray = await boxes.array();
    const scoresArray = await scores.array();

    // Filter out low-confidence threshold
    const threshold = 0.5;
    const result = [];

    for (let i = 0; i < boxesArray[0].length; i++) {
      if (scoresArray[0][i] > threshold) {
        result.push({
          box: boxesArray[0][i],
          score: scoresArray[0][i]
        });
      }
    }

    return result;
  } catch (error) {
    console.error('Error processing predictions:', error);
    return [];
  }
};

export default Camera;