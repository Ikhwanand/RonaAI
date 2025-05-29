import React, { useRef } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import Waves from "../components/Waves";
import womenImage from '../assets/women-2.png'

const FadeInWhenVisible = ({ children, delay = 0 }) => {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
      transition={{ duration: 0.5, delay }}
    >
      {children}
    </motion.div>
  );
};

const HomePage = () => {
  const containerRef = useRef(null);
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
      />
      <div
        ref={containerRef}
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 relative z-10"
      >
        {/* Hero Section */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
        >
          <motion.h1
            className="text-4xl font-bold text-darkText mb-4"
            initial={{ y: -20 }}
            animate={{ y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            AI-Powered Skin Analysis
          </motion.h1>
          <motion.p
            className="text-xl text-lightText max-w-3xl mx-auto mb-8"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.5 }}
          >
            Get personalized skin care recommendations and detect skin issues
            with our advanced AI technology.
          </motion.p>
          {/* <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Link
              to="/analysis"
              className="btn-primary text-lg px-8 py-3 inline-block"
            >
              Start Skin Analysis
            </Link>
          </motion.div> */}
        </motion.div>

        {/* Features Section */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <FadeInWhenVisible delay={0.1}>
            <div className="card text-center">
              <motion.div
                className="bg-primary bg-opacity-10 p-4 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4"
                whileHover={{ rotate: 5, scale: 1.1 }}
                transition={{ type: "spring", stiffness: 400, damping: 10 }}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-8 w-8 text-primary"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                </svg>
              </motion.div>
              <h3 className="text-xl font-semibold mb-2">
                Upload or Take Photos
              </h3>
              <p className="text-lightText">
                Easily upload existing photos or take new ones directly through
                our app.
              </p>
            </div>
          </FadeInWhenVisible>

          <FadeInWhenVisible delay={0.3}>
            <div className="card text-center">
              <motion.div
                className="bg-secondary bg-opacity-10 p-4 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4"
                whileHover={{ rotate: -5, scale: 1.1 }}
                transition={{ type: "spring", stiffness: 400, damping: 10 }}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-8 w-8 text-secondary"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                  />
                </svg>
              </motion.div>
              <h3 className="text-xl font-semibold mb-2">AI Analysis</h3>
              <p className="text-lightText">
                Our AI analyzes your skin condition, detecting issues like acne
                scars and other concerns.
              </p>
            </div>
          </FadeInWhenVisible>

          <FadeInWhenVisible delay={0.5}>
            <div className="card text-center">
              <motion.div
                className="bg-accent bg-opacity-10 p-4 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4"
                whileHover={{ rotate: 5, scale: 1.1 }}
                transition={{ type: "spring", stiffness: 400, damping: 10 }}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-8 w-8 text-accent"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                  />
                </svg>
              </motion.div>
              <h3 className="text-xl font-semibold mb-2">
                Personalized Recommendations
              </h3>
              <p className="text-lightText">
                Get customized skincare advice and treatment recommendations
                based on your analysis.
              </p>
            </div>
          </FadeInWhenVisible>
        </div>

        {/* How It Works Section */}
        <FadeInWhenVisible>
          <div className="mb-16">
            <h2 className="text-3xl font-bold text-center mb-8">
              How It Works
            </h2>
            <div className="flex flex-col md:flex-row justify-between items-center">
              <div className="md:w-1/2 mb-6 md:mb-0 md:pr-8">
                <ol className="space-y-6">
                  {[1, 2, 3, 4].map((step, index) => (
                    <motion.li
                      key={step}
                      className="flex"
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.2, duration: 0.5 }}
                      viewport={{ once: true }}
                    >
                      <motion.span
                        className="bg-gradient-to-r from-primary to-secondary text-white rounded-full w-10 h-10 flex items-center justify-center mr-4 flex-shrink-0 shadow-md"
                        whileHover={{ scale: 1.2, rotate: 5 }}
                        transition={{ type: "spring", stiffness: 300 }}
                      >
                        {step}
                      </motion.span>
                      <div>
                        <h3 className="font-semibold text-lg">
                          {step === 1 && "Upload or Take a Photo"}
                          {step === 2 && "AI Analysis"}
                          {step === 3 && "Get Recommendations"}
                          {step === 4 && "Track Your Progress"}
                        </h3>
                        <p className="text-lightText">
                          {step === 1 &&
                            "Upload an existing photo or use your camera to take a new one."}
                          {step === 2 &&
                            "Our AI analyzes your skin condition and identifies issues."}
                          {step === 3 &&
                            "Receive personalized skincare advice and treatment options."}
                          {step === 4 &&
                            "Monitor improvements over time with follow-up analyses."}
                        </p>
                      </div>
                    </motion.li>
                  ))}
                </ol>
              </div>
              <motion.div
                className="md:w-1/2"
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                viewport={{ once: true }}
              >
                <div className="bg-transparent rounded-lg h-auto flex items-center justify-center overflow-hidden">
                  {/* <motion.p
                    className="text-lightText"
                    animate={{
                      scale: [1, 1.05, 1],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      repeatType: "reverse",
                    }}
                  >
                    App Screenshot Placeholder
                  </motion.p> */}
                  {/* <TiltedCard
                    imageSrc={womenImage}
                    altText="Women's Skin Analysis"
                    captionText="Women's Skin Analysis"
                    containerHeight="auto"
                    containerWidth="100%"
                    imageHeight="auto"
                    imageWidth="100%"
                    rotateAmplitude={12}
                    scaleOnHover={1.1}
                    showMobileWarning={false}
                    showTooltip={true}
                    displayOverlayContent={true}
                    
                  
                  /> */}
                  <img src={womenImage} alt="women" className="rounded-lg h-full w-full object-cover" />
                </div>
              </motion.div>
            </div>
          </div>
        </FadeInWhenVisible>

        {/* CTA Section */}
        <motion.div
          className="bg-primary bg-opacity-10 rounded-xl p-8 text-center"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          viewport={{ once: true }}
        >
          <h2 className="text-2xl font-bold text-darkText mb-4">
            Ready to improve your skin health?
          </h2>
          <p className="text-lg text-lightText mb-6 max-w-2xl mx-auto">
            Start your skin analysis today and get personalized recommendations
            from our AI.
          </p>
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Link
              to="/analysis"
              className="btn-primary text-lg px-8 py-3 inline-block"
            >
              Start Skin Analysis
            </Link>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default HomePage;
