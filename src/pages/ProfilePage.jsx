import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Waves from "../components/Waves";
import ProfileModal from "../components/ProfileModal";
import SkinModal from "../components/SkinModal";
import JournalModal from "../components/JournalModal";
import ProductModal from "../components/ProductModal";
import DeleteAccountModal from "../components/DeleteAccountModal";
import { profileService } from "../services/profileService";
import { skinService } from "../services/skinService";
import { journalService } from "../services/journalService";
import { productService } from "../services/productService";
import { analysisService } from "../services/analysisService";
import { authService } from "../services/authService";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";

const ProfilePage = () => {
  const [activeTab, setActiveTab] = useState("history");
  const [user, setUser] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSkinModalOpen, setIsSkinModalOpen] = useState(false);
  const [skinData, setSkinData] = useState(null);
  const [journalsData, setJournalsData] = useState(null);
  const [productsData, setProductsData] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [isJournalModalOpen, setIsJournalModalOpen] = useState(false);
  const [isProductModalOpen, setIsProductModalOpen] = useState(false);
  const [selectedJournal, setSelectedJournal] = useState(null);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [profile, setProfile] = useState(null);
  const [skinProfile, setSkinProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      navigate("/login");
      return;
    }
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);

      const token = localStorage.getItem("token");
      if (!token) {
        navigate("/login");
        return;
      }

      const [
        profileResponse,
        skinResponse,
        journalsResponse,
        productsResponse,
        analysisResponse,
      ] = await Promise.all([
        profileService.getProfile().catch((err) => {
          console.error("Profile fetch error:", err);
          return { data: null };
        }),
        skinService.getSkinProfile().catch((err) => {
          console.error("Skin profile fetch error:", err);
          return { data: null };
        }),
        journalService.getJournals().catch((err) => {
          console.error("Journals fetch error:", err);
          return { data: { items: [] } };
        }),
        productService.getProducts().catch((err) => {
          console.error("Products fetch error:", err);
          return { data: { items: [] } };
        }),
        analysisService.getAnalysisHistory().catch((err) => {
          console.error("Analysis history fetch error:", err);
          return { data: { items: [] } };
        }),
      ]);

      // console.log("Raw profile response:", profileResponse);

      if (profileResponse?.data) {
        let profileUrl = profileResponse.data.profile_image;
        if (profileUrl && profileUrl !== "null" && profileUrl !== "" && !profileUrl.startsWith("http")) {
          profileUrl = profileUrl.replace(/^\//, "");
          profileUrl = `${import.meta.env.VITE_API_URL}/${profileUrl}`;
        } else if (!profileUrl || profileUrl === "null" || profileUrl === "") {
          profileUrl = null; // Set ke null jika tidak ada gambar profil
        }
        setProfile({
          id: profileResponse.id || profileResponse.data?.id,
          name: profileResponse.name || profileResponse.data?.name,
          email: profileResponse.email || profileResponse.data?.email,
          profileImage: profileUrl,
        });
        setUser(profileResponse.data || profileResponse);
      }

      if (skinResponse?.data) {
        setSkinProfile({
          id: Number(skinResponse.data.id),
          userId: Number(skinResponse.data.user_id),
          skinType: skinResponse.data.skin_type,
          concerns: Array.isArray(skinResponse.data.concerns)
            ? skinResponse.data.concerns
            : [skinResponse.data.concerns],
          createdAt: skinResponse.data.created_at,
          updatedAt: skinResponse.data.updated_at,
        });
        setSkinData(skinResponse.data);
      } else {
        setSkinProfile(null);
        setSkinData(null);
      }

      // console.log("Journals response:", journalsResponse);

      if (journalsResponse?.items) {
        const journalData = journalsResponse.items.map((journal) => ({
          id: journal.id,
          userId: journal.user_id,
          title: journal.title,
          content: journal.content,
          createdAt: journal.created_at,
          updatedAt: journal.updated_at,
        }));
        setJournalsData(journalData);
      } else {
        setJournalsData([]);
      }

      // console.log("Raw Product Response:", productsResponse);

      if (productsResponse?.items) {
        const productData = productsResponse.items.map((product) => ({
          id: product.id,
          userId: product.user_id,
          name: product.product_name,
          category: product.product_category,
          aiRecommendation: product.ai_recommendation,
          createdAt: product.created_at,
          updatedAt: product.updated_at,
          // Add these fields to match the form structure
          product_name: product.product_name,
          product_category: product.product_category,
          ai_recommendation: product.ai_recommendation
        }));
        setProductsData(productData);
      } else {
        setProductsData([]);
      }

      // console.log("Raw analysis response:", analysisResponse);

      if (analysisResponse?.items) {
        const analysisData = analysisResponse.items.map((analysis) => ({
          id: analysis.id,
          userId: analysis.user_id,
          imageUrl: analysis.image_url,
          overallHealth: analysis.overall_health,
          concerns: Array.isArray(analysis.concerns)
            ? analysis.concerns.map((c) => c.name)
            : [],
          recommendations: Array.isArray(analysis.recommendations)
            ? analysis.recommendations.map((r) => r.title)
            : [],
          createdAt: analysis.created_at,
        }));
        setAnalysisData(analysisData);
        // console.log("Analysis items:", analysisData);
      } else {
        console.log("No analysis items found in response", analysisResponse);
        setAnalysisData([]);
      }
    } catch (err) {
      console.error("Error fetching profile data:", err);
      if (err.response?.status === 401) {
        authService.logout();
        navigate("/login");
      } else {
        setError(err.message || "Failed to fetch profile data");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleUpdate = async (updatedData) => {
    try {
      setLoading(true);
      
      // Update local state immediately for better UX
      setUser(prev => ({
        ...prev,
        ...updatedData
      }));
      
      // Refresh all profile data to ensure consistency
      await fetchData();
      
      // No need for additional toast as the individual operations already show toasts
    } catch (error) {
      console.error('Update error:', error);
      toast.error("Failed to refresh profile data");
    } finally {
      setLoading(false);
    }
  };

  const handleSkinUpdate = async (updatedSkin) => {
    try {
      let response;
      if (skinProfile?.id) {
        response = await skinService.updateSkinProfile(updatedSkin);
      } else {
        response = await skinService.createSkinProfile(updatedSkin);
      }

      if (response.success) {
        fetchData();
      }
      return response;
    } catch (error) {
      toast.error("Failed to update skin profile");
      throw error;
    }
  };

  const handleJournalUpdate = async (updatedJournal) => {
    try {
      let response;
      
      if (selectedJournal) {
        // Update existing product
        response = await journalService.updateJournal(selectedJournal.id, updatedJournal);
      } else {
        // Create new product
        response = await journalService.createJournal(updatedJournal);
      }
      
      if (response && response.success) {
        await fetchJournals(); // Refresh the products list
        setIsJournalModalOpen(false);
        setSelectedJournal(null);
      }
      
      return response; // Return the full response object to the modal
    } catch (error) {
      console.error("Error saving journal:", error);
      return { 
        success: false, 
        message: error.message || "Failed to save journal" 
      };
    }
  };

  const handleJournalDelete = async (journalId) => {
    try {
      const response = await journalService.deleteJournal(journalId);
      if (response.success) {
        toast.success("Journal deleted successfully");
        // Refresh products list after successful deletion
        await fetchJournals();
      } else {
        throw new Error(response.message || "Failed to delete journal");
      }
    } catch (err) {
      console.error("Error deleting journal:", err);
      toast.error(err.response?.data?.message || err.message || "Failed to delete journal");
    }
  };

  const fetchJournals = async () => {
    try {
      const response = await journalService.getJournals();
      
      if (response.items) {
        const journalData = response.items.map((journal) => ({
          id: journal.id,
          userId: journal.user_id,
          title: journal.title,
          content: journal.content,
          createdAt: journal.created_at,
          updatedAt: journal.updated_at
          
        }));
        setJournalsData(journalData);
      } else {
        setJournalsData([]);
      }
    } catch (error) {
      console.error('Error fetching journals:', error);
      toast.error('Failed to load journals');
      setProductsData([]);
    }
  };

  const handleProductUpdate = async (updatedProduct) => {
    try {
      let response;
      
      if (selectedProduct) {
        // Update existing product
        response = await productService.updateProduct(selectedProduct.id, updatedProduct);
      } else {
        // Create new product
        response = await productService.createProduct(updatedProduct);
      }
      
      if (response && response.success) {
        await fetchProducts(); // Refresh the products list
        setIsProductModalOpen(false);
        setSelectedProduct(null);
      }
      
      return response; // Return the full response object to the modal
    } catch (error) {
      console.error("Error saving product:", error);
      return { 
        success: false, 
        message: error.message || "Failed to save product" 
      };
    }
  };

  const handleProductDelete = async (productId) => {
    try {
      const response = await productService.deleteProduct(productId);
      if (response.success) {
        toast.success("Product deleted successfully");
        // Refresh products list after successful deletion
        await fetchProducts();
      } else {
        throw new Error(response.message || "Failed to delete product");
      }
    } catch (err) {
      console.error("Error deleting product:", err);
      toast.error(err.response?.data?.message || err.message || "Failed to delete product");
    }
  };

  const fetchProducts = async () => {
    try {
      const response = await productService.getProducts();
      
      if (response.items) {
        const productData = response.items.map((product) => ({
          id: product.id,
          userId: product.user_id,
          name: product.product_name,
          category: product.product_category,
          aiRecommendation: product.ai_recommendation,
          createdAt: product.created_at,
          updatedAt: product.updated_at,
          // Add these fields to match the form structure in ProductModal
          product_name: product.product_name,
          product_category: product.product_category,
          ai_recommendation: product.ai_recommendation
        }));
        setProductsData(productData);
      } else {
        setProductsData([]);
      }
    } catch (error) {
      console.error('Error fetching products:', error);
      toast.error('Failed to load products');
      setProductsData([]);
    }
  };

  const handleDeleteAccount = async () => {
    try {
      const response = await profileService.deleteAccount();
      
      // Check if the response has the expected structure
      if (response && (response.success || (response.data && response.data.success))) {
        // Return true to indicate success to the modal
        return true;
      } else {
        // If response doesn't indicate success, show error
        toast.error("Failed to delete account");
        return false;
      }
    } catch (error) {
      console.error("Error deleting account:", error);
      toast.error(error.response?.data?.message || "An error occurred while deleting your account");
      return false;
    }
  };

  // const handleAnalysisComplete = async (analysisData) => {
  //   try {
  //     // update analysis data
  //     setAnalysisData(prev => [...prev, analysisData.analysis]);

  //     if (analysisData.skinProfile) {
  //       setSkinProfile(prev => ({
  //         ...prev,
  //         skinType: analysisData.skinProfile.skin_type,
  //         concerns: analysisData.skinProfile.concerns
  //       }));

  //       toast.success("Skin profile updated from analysis");
  //     }
  //   } catch (error) {
  //     console.error("Error handling analysis:", error);
  //     toast.error("Failed to update profile from analysis");
  //   }
  // };

  const handleLogout = () => {
    authService.logout();
    navigate("/login");
    toast.success("Logged out successfully");
  };

  // Add loading state handling
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-primary"></div>
      </div>
    );
  }

  // Add error state handling
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-red-600 text-center">
          <h2 className="text-2xl font-bold mb-4">Error</h2>
          <p>{error}</p>
          <button onClick={fetchData} className="mt-4 btn-primary">
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">No Profile Data</h2>
          <p>Unable to load your profile. Please try again.</p>
          <button onClick={fetchData} className="mt-4 btn-primary">
            Retry
          </button>
        </div>
      </div>
    );
  }

  const formatDate = (dateString) => {
    const options = { year: "numeric", month: "short", day: "numeric" };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

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
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h1 className="text-3xl font-bold text-center mb-8">My Profile</h1>

        <div className="grid md:grid-cols-3 gap-8">
          <div className="card md:col-span-1">
            <div className="flex flex-col items-center">
              <div className="w-32 h-32 bg-gray-300 rounded-full mb-4 flex items-center justify-center text-gray-500">
                {profile.profileImage && profile.profileImage !== "null" && profile.profileImage !== "" ? (
                  <img
                    src={profile.profileImage}
                    className="w-32 h-32 rounded-full object-cover"
                    alt="Profile Image"
                  />
                ) : (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-16 w-16"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                    />
                  </svg>
                )}
              </div>

              <h2 className="text-xl font-semibold">{user?.name || "User"}</h2>
              <p className="text-lightText mb-4">{user?.email || "No email"}</p>
              <div className="flex flex-col sm:flex-row w-full gap-2">
                <button
                  className="btn-secondary flex-1"
                  onClick={() => setIsModalOpen(true)}
                >
                  Edit Profile
                </button>
                <button
                  className="btn-primary flex-1"
                  onClick={() => setIsSkinModalOpen(true)}
                >
                  Edit Skin Profile
                </button>
              </div>
            </div>

            <div className="mt-6 pt-6 border-t">
              <h3 className="font-semibold mb-2">Skin Profile</h3>
              <div className="mb-4">
                <p className="text-sm text-lightText">Skin Type:</p>
                <p>{skinProfile?.skinType || "Not set"}</p>
              </div>
              <div>
                <p className="text-sm text-lightText">Primary Concerns:</p>
                <ul className="list-disc list-inside">
                  {skinProfile?.concerns?.map((concern, index) => (
                    <li key={index}>{concern}</li>
                  )) || "Not set"}
                </ul>
              </div>
            </div>

            <div className="mt-6 pt-6 border-t">
              <h3 className="font-semibold mb-2">Quick Stats</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-primary bg-opacity-10 p-3 rounded-lg text-center">
                  <p className="text-2xl font-bold text-primary">
                    {analysisData?.length || 0}
                  </p>
                  <p className="text-xs text-lightText">Analyses</p>
                </div>
                <div className="bg-secondary bg-opacity-10 p-3 rounded-lg text-center">
                  <p className="text-2xl font-bold text-secondary">
                    {journalsData?.length || 0}
                  </p>
                  <p className="text-xs text-lightText">Journal Entries</p>
                </div>
              </div>
            </div>

            <div className="flex flex-col space-y-4 mt-8">
              <button
                onClick={handleLogout}
                className="btn-secondary w-full px-4 py-2 flex items-center justify-center gap-2 rounded"
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
                    d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
                  />
                </svg>
                Logout
              </button>
              <button
                onClick={() => setIsDeleteModalOpen(true)}
                className="w-full px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 flex items-center justify-center gap-2"
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
                Delete Account
              </button>
            </div>
          </div>

          <div className="card md:col-span-2">
            <div className="flex border-b mb-6">
              <button
                className={`px-4 py-2 font-medium ${
                  activeTab === "history"
                    ? "text-primary border-b-2 border-primary"
                    : "text-lightText"
                }`}
                onClick={() => setActiveTab("history")}
              >
                Analysis History
              </button>
              <button
                className={`px-4 py-2 font-medium ${
                  activeTab === "journal"
                    ? "text-primary border-b-2 border-primary"
                    : "text-lightText"
                }`}
                onClick={() => setActiveTab("journal")}
              >
                Skin Journal
              </button>
              <button
                className={`px-4 py-2 font-medium ${
                  activeTab === "products"
                    ? "text-primary border-b-2 border-primary"
                    : "text-lightText"
                }`}
                onClick={() => setActiveTab("products")}
              >
                Saved Products
              </button>
            </div>

            {activeTab === "history" && (
              <>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold">Analysis History</h2>
                  <Link to="/analysis" className="btn-primary">
                    New Analysis
                  </Link>
                </div>

                {analysisData && analysisData.length > 0 ? (
                  <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                    {analysisData.map((analysis) => {
                      const safeSlug = (
                        analysis.skinType || "unknown"
                      ).toLowerCase();
                      return (
                        <div
                          key={analysis.id}
                          className="flex items-center p-4 border rounded-lg hover:bg-gray-50"
                        >
                          <div className="flex-grow">
                            <p className="font-medium">
                              {formatDate(analysis.createdAt)}
                            </p>
                            <div className="flex space-x-4 text-sm text-lightText">
                              <p>Health: {analysis.overallHealth}</p>
                              <div>
                                Concerns:{" "}
                                {analysis.concerns &&
                                analysis.concerns.length > 0 ? (
                                  analysis.concerns.map((concern, i) => (
                                    <span key={i} className="mr-2">
                                      {concern}
                                    </span>
                                  ))
                                ) : (
                                  <span>No concerns detected</span>
                                )}
                              </div>
                            </div>
                          </div>
                          <Link
                            to={`/results/${analysis.id}/${safeSlug}`}
                            className="text-primary hover:underline"
                          >
                            View Details
                          </Link>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p className="text-lightText mb-4">
                      You haven't done any skin analysis yet.
                    </p>
                    <Link to="/analysis" className="btn-primary">
                      Start Your First Analysis
                    </Link>
                  </div>
                )}
              </>
            )}

            {activeTab === "journal" && (
              <>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold">Skin Journal</h2>
                  <button
                    className="btn-primary"
                    onClick={() => setIsJournalModalOpen(true)}
                  >
                    Add Entry
                  </button>
                </div>

                {journalsData && journalsData.length > 0 ? (
                  <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                    {journalsData?.map((entry) => (
                      <div key={entry.id} className="p-4 border rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                          <h3 className="font-medium">{entry.title}</h3>
                          <div className="flex gap-2">
                          <button
                            onClick={() => {
                              setSelectedJournal(entry);
                              setIsJournalModalOpen(true);
                            }}
                            className="text-primary hover:text-primary-dark p-1 rounded-full hover:bg-gray-100"
                            title="Edit Journal"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                            </svg>
                          </button>
                          <button
                            onClick={() => handleJournalDelete(entry.id)}
                            className="text-red-600 hover:text-red-800 p-1 rounded-full hover:bg-gray-100"
                            title="Delete Journal"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                          </div>
                        </div>
                        <p className="text-lightText">{entry.content}</p>
                        <p className="text-sm text-gray-500 mt-2">
                          {formatDate(entry.createdAt)}
                        </p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p className="text-lightText mb-4">
                      You haven't added any journal entries yet.
                    </p>
                    <button
                      className="btn-primary"
                      onClick={() => setIsJournalModalOpen(true)}
                    >
                      Create Your First Entry
                    </button>
                  </div>
                )}

                <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                  <h3 className="font-medium mb-2">Why Keep a Skin Journal?</h3>
                  <p className="text-lightText">
                    Tracking your skincare routine, diet, and lifestyle changes
                    can help identify what works best for your skin. Our AI uses
                    this information to provide more personalized
                    recommendations.
                  </p>
                </div>
              </>
            )}

            {activeTab === "products" && (
              <>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold">Saved Products</h2>
                  <button
                    className="btn-primary"
                    onClick={() => {
                      setSelectedProduct(null);
                      setIsProductModalOpen(true);
                    }}
                  >
                    Add Product
                  </button>
                </div>

                {productsData && productsData.length > 0 ? (
                  <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                    {productsData?.map((product) => (
                      <div
                        key={product.id}
                        className="flex items-center p-4 border rounded-lg"
                      >
                        <div className="w-12 h-12 bg-gray-200 rounded-lg mr-4 flex items-center justify-center">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-6 w-6 text-gray-500"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                            />
                          </svg>
                        </div>
                        <div className="flex-grow">
                          <p className="font-medium">{product.name}</p>
                          <p className="text-sm text-lightText">
                            Category: {product.category}
                          </p>
                        </div>
                        {product.aiRecommendation && (
                          <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                            AI Recommended
                          </span>
                        )}
                        <div className="flex gap-2 ml-4">
                          <button
                            onClick={() => {
                              setSelectedProduct(product);
                              setIsProductModalOpen(true);
                            }}
                            className="text-primary hover:text-primary-dark p-1 rounded-full hover:bg-gray-100"
                            title="Edit Product"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                            </svg>
                          </button>
                          <button
                            onClick={() => handleProductDelete(product.id)}
                            className="text-red-600 hover:text-red-800 p-1 rounded-full hover:bg-gray-100"
                            title="Delete Product"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p className="text-lightText mb-4">
                      You haven't saved any products yet.
                    </p>
                    <button
                      className="btn-primary"
                      onClick={() => setIsProductModalOpen(true)}
                    >
                      Add Your First Product
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
      <ProfileModal
        user={user}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onUpdate={handleUpdate}
      />
      <SkinModal
        skin={skinProfile || { skin_type: "", concerns: "" }}
        isOpen={isSkinModalOpen}
        onClose={() => setIsSkinModalOpen(false)}
        onUpdate={handleSkinUpdate}
      />
      <JournalModal
        entry={selectedJournal}
        isOpen={isJournalModalOpen}
        onClose={() => {
          setIsJournalModalOpen(false);
          setSelectedJournal(null);
        }}
        onUpdate={handleJournalUpdate}
      />
      <ProductModal
        product={selectedProduct}
        isOpen={isProductModalOpen}
        onClose={() => {
          setIsProductModalOpen(false);
          setSelectedProduct(null);
        }}
        onUpdate={handleProductUpdate}
      />
      <DeleteAccountModal
        isOpen={isDeleteModalOpen}
        onClose={() => setIsDeleteModalOpen(false)}
        onDelete={handleDeleteAccount}
      />
    </div>
  );
};

export default ProfilePage;
