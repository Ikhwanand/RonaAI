import React, { useState, useEffect } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { analysisService } from "../services/analysisService";
import Waves from "../components/Waves";
import { toast } from "react-toastify";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import ProgressChart from "../components/ProgressChart";
import ConcernsProgress from "../components/ConcernsProgress";
import SkincareRecommendations from "../components/SkincareRecommendations";

const ResultsPage = () => {
  const [activeTab, setActiveTab] = useState("assessment");
  const [analysisData, setAnalysisData] = useState(null);
  const [progressData, setProgressData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const { id, slug } = useParams();

  useEffect(() => {
    fetchAnalaysisData();
  }, [id]);

  const fetchAnalaysisData = async () => {
    try {
      setLoading(true);
      const [analysisResponse, progressResponse] = await Promise.all([
        analysisService.getAnalysis(id),
        analysisService.getProgress(id).catch(err => {
          console.log("Progress data error:", err);
          return { success: false, message: "Could not fetch progress data" };
        }),
      ]);

      if (analysisResponse.success) {
        let imageUrl = analysisResponse.data.image_url;
        if (!imageUrl.startsWith("http")) {
          imageUrl = imageUrl.replace(/^\//, "");
          imageUrl = `${import.meta.env.VITE_API_URL}/${imageUrl}`;
        }

        setAnalysisData({
          id: analysisResponse.data.id,
          imageUrl: imageUrl,
          overallHealth: analysisResponse.data.overall_health,
          skinType: analysisResponse.data.skin_type || "unknown",
          concerns:
            analysisResponse.data.concerns?.map((concern) => ({
              name: concern.name,
              severity: concern.severity,
              type: concern.type,
              confidence: concern.confidence,
            })) || [],
          recommendations:
            analysisResponse.data.recommendations?.map((rec) => ({
              title: rec.title,
              description: rec.description,
              priority: rec.priority,
            })) || [],
          analysisMetrics: analysisResponse.data.analysis_metrics || {
            skin_hydration: 0,
            texture_uniformity: 0,
            pore_visibility: 0,
            overall_score: 0,
          },
          skincareRecommendations: analysisResponse.data.skincare_products || [],
          createdAt: analysisResponse.data.created_at,
        });
      }

      console.log("Progress response:", progressResponse);

      if (progressResponse.success) {
        setProgressData({
          isFirstAnalysis: progressResponse.data.is_first_analysis || false,
          comparisonDate: progressResponse.data.comparison_date,
          improvementAreas: progressResponse.data.improvement_areas || [],
          metricsComparison: progressResponse.data.metrics_comparison || {},
          concernsProgress: progressResponse.data.concerns_progress || [],
          currentMetrics: progressResponse.data.current_metrics || {}
        });
      } else {
        // Set default data for first analysis
        setProgressData({
          isFirstAnalysis: true,
          currentMetrics: analysisData?.
          analysisMetrics || {
            skin_hydration: 0,
            texture_uniformity: 0,
            pore_visibility: 0,
            overall_score: 0
          }
        });
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      setError(error.message || "Failed to load analysis data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (analysisData && slug) {
      const skinType = analysisData.skinType || "unknown";
      const expectedSlug =
        typeof skinType === "string" ? skinType.toLowerCase() : "unknown";

      if (expectedSlug && slug !== expectedSlug) {
        navigate(`/results/${id}/${expectedSlug}`, { replace: true });
      }
    }
  }, [analysisData, slug, id, navigate]);

  // Mock data - in a real app, you would fetch this from your API
  const handleDeleteAnalysis = async (analysisId) => {
    try {
      const response = await analysisService.deleteAnalysis(analysisId);
      if (response.success) {
        navigate("/profile");
      }
    } catch (error) {
      setError("Failed to delete analysis");
    }
  };

  const handleExportPDF = async () => {
    try {
      // Create a container to render the report content
      const reportContainer = document.createElement("div");
      reportContainer.style.padding = "40px";
      reportContainer.style.position = "absolute";
      reportContainer.style.left = "-9999px";
      reportContainer.style.top = "0"; // Changed from -9999px to 0
      reportContainer.style.width = "595px";
      reportContainer.style.fontFamily = "'Times New Roman', Times, serif";
      reportContainer.style.overflow = "visible";

      // Create formal report content
      reportContainer.innerHTML = `
        <div style="color: #333;">
          <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-size: 24px; margin-bottom: 5px; font-weight: bold;">Skin Analysis Report</h1>
            <p style="font-size: 14px; color: #555;">Generated by RonaAI</p>
            <div style="border-bottom: 2px solid #333; width: 100px; margin: 15px auto;"></div>
          </div>
          
          <div style="margin-bottom: 30px; text-align: center;">
            <p style="font-size: 14px; margin-bottom: 5px;">Date: ${new Date(
              analysisData.createdAt
            ).toLocaleDateString("en-US", {
              year: "numeric",
              month: "long",
              day: "numeric",
            })}</p>
            <p style="font-size: 14px;">Analysis ID: ${analysisData.id}</p>
          </div>
          
          <div style="margin-bottom: 30px; text-align: center;">
            <img src="${analysisData.imageUrl}" 
                 alt="Skin Analysis" 
                 style="max-width: 80%; border: 1px solid #ddd; border-radius: 4px;"
                 crossorigin="anonymous" />
          </div>
          
          <div style="margin-bottom: 30px;">
            <h2 style="font-size: 18px; border-bottom: 1px solid #333; padding-bottom: 5px; margin-bottom: 15px;">
              Analysis Summary
            </h2>
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
              <p style="font-weight: bold;">Overall Health:</p>
              <p>${analysisData.overallHealth}</p>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
              <p style="font-weight: bold;">Skin Type:</p>
              <p>${analysisData.skinType}</p>
            </div>
          </div>
          
          <div style="margin-bottom: 30px;">
            <h2 style="font-size: 18px; border-bottom: 1px solid #333; padding-bottom: 5px; margin-bottom: 15px;">
              Detected Skin Concerns
            </h2>
            <div id="concerns-container"></div>
          </div>
          
          <div style="margin-bottom: 30px;">
            <h2 style="font-size: 18px; border-bottom: 1px solid #333; padding-bottom: 5px; margin-bottom: 15px;">
              Professional Recommendations
            </h2>
            <div id="recommendations-container"></div>
          </div>
          
          <div style="margin-top: 40px; text-align: center; font-size: 12px; color: #777;">
            <p>This report was generated automatically by RonaAI.</p>
            <p>For professional medical advice, please consult a dermatologist.</p>
          </div>
        </div>
      `;

      document.body.appendChild(reportContainer);

      // Add concerns with formal styling
      const concernsContainer = reportContainer.querySelector(
        "#concerns-container"
      );
      analysisData.concerns.forEach((concern) => {
        const concernDiv = document.createElement("div");
        concernDiv.style.marginBottom = "15px";
        concernDiv.style.padding = "10px";
        concernDiv.style.border = "1px solid #eee";
        concernDiv.style.borderRadius = "4px";
        concernDiv.innerHTML = `
          <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <p style="font-weight: bold;">${concern.name}</p>
            <p style="color: ${
              concern.severity.toLowerCase() === "severe"
                ? "#d32f2f"
                : concern.severity.toLowerCase() === "moderate"
                ? "#ffa000"
                : "#388e3c"
            }">${concern.severity}</p>
          </div>
          ${
            concern.type
              ? `<p style="font-size: 13px; margin-bottom: 5px;">Type: ${concern.type}</p>`
              : ""
          }
          <p style="font-size: 13px;">Confidence Level: ${(
            concern.confidence * 100
          ).toFixed(1)}%</p>
        `;
        concernsContainer.appendChild(concernDiv);
      });

      // Add recommendations with formal styling
      const recommendationsContainer = reportContainer.querySelector(
        "#recommendations-container"
      );
      analysisData.recommendations.forEach((rec, i) => {
        const recDiv = document.createElement("div");
        recDiv.style.marginBottom = "20px";
        recDiv.style.padding = "15px";
        recDiv.style.backgroundColor = "#f9f9f9";
        recDiv.style.borderLeft = `4px solid ${
          rec.priority === "High"
            ? "#d32f2f"
            : rec.priority === "Medium"
            ? "#ffa000"
            : "#388e3c"
        }`;
        recDiv.innerHTML = `
          <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: bold; margin-right: 10px;">${
              i + 1
            }.</span>
            <h3 style="font-weight: bold; margin: 0; flex-grow: 1; bottom: 2px;">${
              rec.title
            }</h3>
            <span style="font-size: 13px; padding: 2px 8px; 
              background-color: ${
                rec.priority === "High"
                  ? "#ffebee"
                  : rec.priority === "Medium"
                  ? "#fff8e1"
                  : "#e8f5e9"
              };
              color: ${
                rec.priority === "High"
                  ? "#d32f2f"
                  : rec.priority === "Medium"
                  ? "#ffa000"
                  : "#388e3c"
              };
              border-radius: 4px;
              align-self: center;">
              ${rec.priority} Priority
            </span>
          </div>
          <p style="font-size: 14px; line-height: 1.5;">${rec.description}</p>
        `;
        recommendationsContainer.appendChild(recDiv);
      });

      // Load image first to ensure it's available
      const img = reportContainer.querySelector("img");
      await new Promise((resolve) => {
        if (img.complete) {
          resolve();
        } else {
          img.onload = resolve;
          img.onerror = () => {
            console.error("Error loading image for PDF");
            resolve();
          };
        }
      });

      await new Promise((resolve) => setTimeout(resolve, 500));

      // Use html2canvas to capture the entire report
      const canvas = await html2canvas(reportContainer, {
        useCORS: true,
        scale: 2,
        logging: false,
        backgroundColor: "#ffffff",
        scrollX: 0,
        scrollY: -window.scrollY,
        windowHeight: reportContainer.scrollHeight,
        allowTaint: true,
      });

      // Create PDF
      const imgData = canvas.toDataURL("image/jpeg", 1.0);
      const pdf = new jsPDF("p", "pt", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();
      // const ratio = canvas.width / canvas.height;
      const imgWidth = pdfWidth;
      const imgHeight = (canvas.height * pdfWidth) / canvas.width;

      // Add content to PDF with proper scaling
      let heightLeft = imgHeight;
      let position = 0;

      pdf.addImage(imgData, "JPEG", 0, position, imgWidth, imgHeight);
      heightLeft -= pdfHeight;

      // Add new pages as needed
      while (heightLeft >= 0) {
        position = heightLeft - imgHeight;
        pdf.addPage();
        pdf.addImage(imgData, "JPEG", 0, position, imgWidth, imgHeight);
        heightLeft -= pdfHeight;
      }

      // Clean up
      document.body.removeChild(reportContainer);

      // Save the PDF
      const filename = `skin-analysis-${analysisData.id}.pdf`;
      pdf.save(filename);

      toast.success("PDF exported successfully");
    } catch (error) {
      console.error("Error generating PDF:", error);
      toast.error("Failed to generate PDF");
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity.toLowerCase()) {
      case "severe":
        return "text-red-600";
      case "moderate":
        return "text-yellow-600";
      case "mild":
        return "text-green-600";
      default:
        return "text-gray-600";
    }
  };

  const renderProgressTab = () => {
    if (loading) {
      return (
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="text-center py-8 text-red-500">
          <p className="mb-4">{error}</p>
          <button className="btn-primary px-6 py-2" onClick={fetchAnalaysisData}>
            Try Again
          </button>
        </div>
      );
    }

    if (!progressData) {
      return (
        <div className="text-center py-8">
          <p className="mb-4">No progress data available.</p>
          <p>This appears to be your first skin analysis. Complete more analyses over time to track your progress!</p>
        </div>
      );
    }

    if (progressData.isFirstAnalysis) {
      return (
        <div className="text-center py-8">
          <h3 className="text-xl font-semibold mb-4">First Analysis</h3>
          <p className="mb-6">This is your first skin analysis. Complete more anlyses over time to track your progress!</p>

          <div className="bg-white rounded-lg shadow-md p-6 mb-6 max-w-md mx-auto">
            <h4 className="text-lg font-medium mb-4">Current Metrics</h4>
            {Object.entries(progressData.currentMetrics || {}).map(([key, value]) => (
              <div key={key} className="flex justify-between py-2 border-b">
                <span className="capitalize">{key.replace(/_/g, ' ')}</span>
                <span className="font-medium">{typeof value === 'number' ? value.toFixed(1) : value}</span>
              </div>
            ))}
          </div>
        </div>
      );
    }

    return (
      <div className="py-6">
        <div className="mb-6">
          <h3 className="text-xl font-semibold mb-2">Progress Overview</h3>
          {progressData.comparisonDate && (
            <p className="text-gray-600">
              Comparing with analysis from {progressData.comparisonDate} 
            </p>
          )}
        </div>

        {progressData.improvementAreas && progressData.improvementAreas.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h3 className="text-xl font-semibold mb-4">Areas of Improvement</h3>
            <div className="flex flex-wrap gap-2">
              {progressData.improvementAreas.map((area) => (
                <span key={area}
                  className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm"
                >
                  {area}
                </span>
              ))}
            </div>
          </div>
        )}

        <ProgressChart metricsComparison={progressData.metricsComparison} />
        <ConcernsProgress concernsProgress={progressData.concernsProgress} />

        {/* <div className="mt-8 text-center">
          <button className="btn-primary px-6 py-2"
            onClick={() => {
              const reportElement = document.getElementById('progress-report');
              if (reportElement) {
                html2canvas(reportElement).then((canvas) => {
                  const imgData = canvas.toDataURL('image/png');
                  const pdf = new jsPDF('p', 'mm', 'a4');
                  const pdfWidth = pdf.internal.pageSize.getWidth();
                  const pdfHeight = pdf.internal.pageSize.getHeight();
                  const imgWidth = canvas.width;
                  const imgHeight = canvas.height;
                  const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight);
                  const imgX = (pdfWidth - imgWidth * ratio) / 2;
                  const imgY = 30;

                  pdf.addImage(imgData, 'PNG', imgX, imgY, imgWidth * ratio, imgHeight * ratio);
                  pdf.save('progress-report.pdf');
                });
              } else {
                toast.error('Could not generate PDF report');
              }
            }}
          >
            Download Progress Report
          </button>
        </div> */}
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-red-600 text-center">
          <h2 className="text-2xl font-bold mb-4">Error</h2>
          <p>{error}</p>
          <button onClick={fetchAnalaysisData} className="mt-4 btn-primary">
            Retry
          </button>
        </div>
      </div>
    );
  }

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

      {analysisData && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <h1 className="text-3xl font-bold text-center mb-8">
            Analysis Results
          </h1>

          <div className="flex flex-wrap border-b mb-8 overflow-x-auto">
            <button
              className={`px-3 py-2 text-sm md:text-base md:px-4 md:py-2 font-medium whitespace-nowrap ${
                activeTab === "assessment"
                  ? "text-primary border-b-2 border-primary"
                  : "text-lightText hover:text-gray-700"
              }`}
              onClick={() => setActiveTab("assessment")}
            >
              Assessment
            </button>
            <button
              className={`px-3 py-2 text-sm md:text-base md:px-4 md:py-2 font-medium whitespace-nowrap ${
                activeTab === "metrics"
                  ? "text-primary border-b-2 border-primary"
                  : "text-lightText hover:text-gray-700"
              }`}
              onClick={() => setActiveTab("metrics")}
            >
              Metrics
            </button>
            <button
              className={`px-3 py-2 text-sm md:text-base md:px-4 md:py-2 font-medium whitespace-nowrap ${
                activeTab === "progress"
                  ? "text-primary border-b-2 border-primary"
                  : "text-lightText hover:text-gray-700"
              }`}
              onClick={() => setActiveTab("progress")}
            >
              Progress
            </button>
            <button
              className={`px-3 py-2 text-sm md:text-base md:px-4 md:py-2 font-medium whitespace-nowrap ${
                activeTab === "skincareRecommendations"
                 ? "text-primary border-b-2 border-primary"
                 : "text-lightText hover:text-gray-700"
              }`}
              onClick={() => setActiveTab("skincareRecommendations")}
            >
              Skincare Recommendations
            </button>
          </div>

          {activeTab === "assessment" && (
            <>
              <div className="grid md:grid-cols-2 gap-8 mb-8">
                <div className="card">
                  <h2 className="text-xl font-semibold mb-4">
                    Your Skin Image
                  </h2>
                  <img
                    src={analysisData.imageUrl}
                    alt="Analyzed skin"
                    className="rounded-lg w-full"
                    // onError={(e) => {
                    //   e.target.onerror = null;
                    //   e.target.src = '/placeholder-image.jpg';
                    // }}
                  />
                </div>

                <div className="card">
                  <h2 className="text-xl font-semibold mb-4">
                    Skin Assessment
                  </h2>
                  <div className="mb-4">
                    <p className="text-lightText mb-1">Overall Health:</p>
                    <p className="font-medium text-lg">
                      {analysisData.overallHealth}
                    </p>
                  </div>

                  <div className="mb-4">
                    <p className="text-lightText mb-1">Skin Type:</p>
                    <p className="font-medium text-lg">
                      {analysisData.skinType}
                    </p>
                  </div>

                  <div>
                    <p className="text-lightText mb-2">Detected Concerns:</p>
                    {analysisData.concerns.map((concern, index) => (
                      <div key={index} className="mb-2 last:mb-0">
                        <div className="flex justify-between">
                          <p className="font-medium">{concern.name}</p>
                          <p
                            className={`font-medium ${getSeverityColor(
                              concern.severity
                            )}`}
                          >
                            {concern.severity}
                          </p>
                        </div>
                        {concern.type && (
                          <p className="text-sm text-lightText">
                            Type: {concern.type}
                          </p>
                        )}
                        <p className="text-sm text-lightText">
                          Confidence: {(concern.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="card mb-8">
                <h2 className="text-xl font-semibold mb-4">Recommendations</h2>
                <div className="space-y-4">
                  {analysisData.recommendations.map((rec, index) => (
                    <div
                      key={index}
                      className="pb-4 border-b last:border-b-0 last:pb-0"
                    >
                      <div className="flex items-center gap-2">
                        <h3 className="font-medium text-lg">
                          {index + 1}. {rec.title}
                        </h3>
                        <span
                          className={`text-sm px-2 py-1 rounded ${
                            rec.priority === "High"
                              ? "bg-red-100 text-red-800"
                              : rec.priority === "Medium"
                              ? "bg-yellow-100 text-yellow-800"
                              : "bg-green-100 text-green-800"
                          }`}
                        >
                          {rec.priority}
                        </span>
                      </div>
                      <p className="text-lightText mt-1">{rec.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {activeTab === "metrics" && (
            <div className="card mb-8">
              <h2 className="text-xl font-semibold mb-4">Analysis Metrics</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(analysisData.analysisMetrics).map(
                  ([key, value]) => (
                    <div
                      key={key}
                      className="text-center p-4 bg-gray-50 rounded-lg"
                    >
                      <p className="text-lightText text-sm mb-1">
                        {key
                          .split("_")
                          .map(
                            (word) =>
                              word.charAt(0).toUpperCase() + word.slice(1)
                          )
                          .join(" ")}
                      </p>
                      <p className="text-2xl font-bold">{value}%</p>
                    </div>
                  )
                )}
              </div>
            </div>
          )}

          {activeTab === "progress" && renderProgressTab(
            // <div className="card mb-8">
            //   <h2 className="text-xl font-semibold mb-4">Progress Tracking</h2>
            //   <div className="mb-6">
            //     <p className="text-lg font-medium mb-2">Overall Improvement</p>
            //     <p className="text-3xl font-bold text-primary">
            //       {progressData.overallImprovement
            //         ? (progressData.overallImprovement * 100).toFixed(1)
            //         : 0}
            //       %
            //     </p>
            //   </div>

            //   <div className="space-y-6">
            //     {progressData.length > 1 ? (
            //       progressData.concernsProgress.map((concern, index) => (
            //         <div key={index} className="border-b pb-4 last:border-b-0">
            //           <p className="font-medium mb-2">
            //             {concern.concern_name || "Unknown Concern"}
            //           </p>
            //           <div className="flex items-center gap-4">
            //             <span
            //               className={getSeverityColor(
            //                 concern.previous_severity || "Unknown"
            //               )}
            //             >
            //               {concern.previous_severity || "Unknown"}
            //             </span>
            //             <span className="text-gray-400">→</span>
            //             <span
            //               className={getSeverityColor(
            //                 concern.current_severity || "Unknown"
            //               )}
            //             >
            //               {concern.current_severity || "Unknown"}
            //             </span>
            //           </div>
            //           <p className="text-sm text-gray-500 mt-1">
            //             Improvement:{" "}
            //             {(concern.improvement_percentage || 0).toFixed(1)}%
            //           </p>
            //         </div>
            //       ))
            //     ) : (
            //       <p className="text-gray-500">
            //         {progressData.length === 0
            //           ? "No progress data"
            //           : "Not enough data to compare"}
            //       </p>
            //     )}
            //   </div>
            // </div>
          )}

          {activeTab === "skincareRecommendations" && (
            <div className="mb-8">
              <SkincareRecommendations products={analysisData.skincareRecommendations} />

            </div>
          )}

          <div className="flex justify-center space-x-4">
            <Link
              to="/analysis"
              className="btn-secondary flex items-center gap-2"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                />
              </svg>
              New Analysis
            </Link>
            <button
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md font-medium transition-colors flex items-center gap-2"
              onClick={() => handleDeleteAnalysis(analysisData.id)}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
              Delete Analysis
            </button>
            <button
              className="btn-secondary flex items-center"
              onClick={handleExportPDF}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5 mr-2"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6zm5 6a1 1 0 10-2 0v3.586l-1.293-1.293a1 1 0 10-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 11.586V8z"
                  clipRule="evenodd"
                />
              </svg>
              Export PDF
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsPage;
