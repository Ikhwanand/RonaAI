import React from "react";
import { motion } from "framer-motion";

const SkincareRecommendations = ({ products }) => {
  if (!products || products.length === 0) {
    return (
      <div className="p-4 text-center">
        <p className="text-gray-500">No product recommendations available.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold mb-4 text-center md:text-left">
        Skincare Product Recommendations
      </h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 md:gap-6">
        {products.map((product, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100"
          >
            <div className="p-3 md:p-5">
              <div className="flex flex-col sm:flex-row justify-between items-start mb-3 gap-2">
                <h4 className="text-lg font-semibold break-words">{product.title}</h4>
                <span
                  className={`px-2 py-1 text-xs rounded-full whitespace-nowrap ${getPriorityBadgeColor(
                    product.priority
                  )}`}
                >
                  {product.priority}
                </span>
              </div>

              <p className="text-gray-600 mb-4 text-sm md:text-base">{product.description}</p>

              <div className="mb-4">
                <span className="text-base md:text-lg font-medium text-indigo-600">
                  {product.price}
                </span>
              </div>

              <div className="space-y-2 md:space-y-3 mb-4">
                <DetailItem label="How to use" value={product.how_to_use} />
                <DetailItem label="Benefits" value={product.benefits} />
                <DetailItem
                  label="Side effect"
                  value={product.side_effects}
                />
                <DetailItem label="Dosage" value={product.dosage} />
              </div>

              {/* <a
                href={product.link.replace(/`/g, "")}
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full text-center bg-indigo-600 hover:bg-indigo-700 text-white py-2 px-4 rounded-md transition duration-300"
              >
                See Product
              </a> */}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

const DetailItem = ({ label, value }) => (
  <div>
    <span className="text-sm font-medium text-gray-500">{label}:</span>
    <p className="text-gray-700 text-sm md:text-base">{value}</p>
  </div>
);

const getPriorityColor = (priority) => {
  switch (priority.toLowerCase()) {
    case "high":
      return "bg-red-500";
    case "medium":
      return "bg-yellow-500";
    case "low":
      return "bg-green-500";
    default:
      return "bg-gray-500";
  }
};

const getPriorityBadgeColor = (priority) => {
  switch (priority.toLowerCase()) {
    case "high":
      return "bg-red-100 text-red-800";
    case "medium":
      return "bg-yellow-100 text-yellow-800";
    case "low":
      return "bg-green-100 text-green-800";
    default:
      return "bg-gray-100 text-gray-800";
  }
};

export default SkincareRecommendations;
