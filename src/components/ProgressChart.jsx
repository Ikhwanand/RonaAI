import React from "react";
import { motion } from "framer-motion";

const ProgressChart = ({ metricsComparison }) => {
  if (!metricsComparison || Object.keys(metricsComparison).length === 0) {
    return (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h3 className="text-xl font-semibold mb-4">Metrics Progress</h3>
            <div className="text-gray-500">No metrics data available for comparison.</div>
        </div>
    );
  };

  const metrics = [
    { key: "skin_hydration", label: "Skin Hydration" },
    { key: "texture_uniformity", label: "Texture Uniformity" },
    { key: "pore_visibility", label: "Pore Visibility" },
    { key: "overall_score", label: "Overall Score" },
  ];

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-xl font-semibold mb-4">Metrics Progress</h3>

        <div className="space-y-6">
            {metrics.map((metric) => {
                const data = metricsComparison[metric.key];
                if (!data) return null;

                const previousValue = data.previous || 0;
                const currentValue = data.current || 0;
                const improved = data.improved;

                // Calculate percentage for progress bar
                const maxValue = 100;
                const previousPercent = (previousValue / maxValue) * 100;
                const currentPercent = (currentValue / maxValue) * 100;

                return (
                    <div key={metric.key} className="space-y-2">
                        <div className="flex justify-between items-center">
                            <span className="font-medium">{metric.label}</span>
                            <span className={`text-sm font-medium ${improved ? 'text-green-600' : 'text-red-600'}`}>
                                {improved ? '↑' : '↓'}{" "}{Math.abs(currentValue - previousValue).toFixed(1)}
                            </span>
                        </div>

                        <div className="relative h-8 bg-gray-200 rounded-full overflow-hidden">
                            {/* Previous value bar */}
                            <div className="absolute h-full bg-indigo-300 opacity-50 rounded-full" style={{ width: `${previousPercent}%` }} />

                            {/* Current value bar with animation */}
                            <motion.div
                                className="absolute h-full bg-indigo-600 rounded-full"
                                initial={{ width: 0 }}
                                animate={{ width: `${currentPercent}%` }}
                                transition={{ duration: 1, ease: "easeOut" }}
                            />

                            {/* Labels */}
                            <div className="absolute inset-0 flex items-center justify-between px-3 text-xs text-white font-medium">
                                <span>Previous: {previousValue.toFixed(1)}</span>
                                <span>Current: {currentValue.toFixed(1)}</span>
                            </div>
                        </div>
                    </div>
                );
            })}
        </div>
    </div>
  );
};

export default ProgressChart;
