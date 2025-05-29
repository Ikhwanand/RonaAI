import React from "react";
import { motion } from "framer-motion";

const ConcernsProgress = ({ concernsProgress }) => {
  if (!concernsProgress || concernsProgress.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-xl font-semibold mb-4">Skin Concerns</h3>
        <p className="text-gray-500">No skin concerns data available.</p>
      </div>
    );
  }

  // Helper function to convert severity string to number
  const getSeverityValue = (severity) => {
    if (typeof severity === 'number') return severity;
    
    const severityMap = {
      'Severe': 8,
      'Moderate': 5,
      'Mild': 3,
      'None': 0
    };
    
    return severityMap[severity] || 0;
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <h3 className="text-xl font-semibold mb-4">Skin Concerns Progress</h3>

      <div className="space-y-4">
        {concernsProgress.map((concern, index) => {
          const isNew = concern.is_new;
          const name = concern.name;
          const currentSeverity = getSeverityValue(concern.current_severity);
          const previousSeverity = getSeverityValue(concern.previous_severity);
          const improved = !isNew && currentSeverity < previousSeverity;

          // Calculate severity percentage (scale 0-10)
          const maxSeverity = 10;
          const currentPercent = (currentSeverity / maxSeverity) * 100;
          const previousPercent = isNew
            ? 0
            : (previousSeverity / maxSeverity) * 100;

          return (
            <motion.div
              key={`${name}-${index}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="border rounded-lg p-4"
            >
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium">{name}</span>
                {isNew ? (
                  <span className="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded">
                    New
                  </span>
                ) : (
                  <span
                    className={`text-sm ${
                      improved ? "text-green-600" : "text-red-600"
                    } font-medium`}
                  >
                    {improved ? "Improved" : "Worsened"}
                  </span>
                )}
              </div>

              {!isNew && (
                <div className="relative h-6 bg-gray-200 rounded-full overflow-hidden mb-1">
                  {/* Previous severity bar */}
                  <div
                    className="absolute h-full bg-orange-300 opacity-50 rounded-full"
                    style={{ width: `${previousPercent}%` }}
                  />

                  {/* Current severity bar */}
                  <motion.div
                    className="absolute h-full bg-orange-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${currentPercent}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />

                  <div className="absolute inset-0 flex items-center px-3">
                    <span className="text-xs text-white font-medium">
                      Severity: {typeof concern.current_severity === 'string' ? concern.current_severity : `${currentSeverity}/10`}
                    </span>
                  </div>
                </div>
              )}

              {isNew && (
                <div className="relative h-6 bg-gray-200 rounded-full overflow-hidden mb-1">
                  <motion.div
                    className="absolute h-full bg-orange-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${currentPercent}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />

                  <div className="absolute inset-0 flex items-center px-3">
                    <span className="text-xs text-white font-medium">
                      Severity: {typeof concern.current_severity === 'string' ? concern.current_severity : `${currentSeverity}/10`}
                    </span>
                  </div>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};

export default ConcernsProgress;
