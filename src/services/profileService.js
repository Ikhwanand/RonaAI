import api from "../utils/api";

export const profileService = {
  getProfile: async () => {
    try {
      const response = await api.get("/profile/me");
      console.log("Raw profile response:", response || response.data);
      return response || response.data;
    } catch (error) {
      console.error("Error in profileService:", error);
      throw error;
    }
  },

  updateProfile: async (profileData) => {
    try {
      // Filter out empty fields to avoid validation errors
      const filteredData = Object.fromEntries(
        Object.entries(profileData).filter(
          ([_, v]) => v !== null && v !== undefined && v !== ""
        )
      );

      // Only send the request if there are fields to update
      if (Object.keys(filteredData).length === 0) {
        return { success: true, message: "No changes to update" };
      }

      const response = await api.put("/profile/update", filteredData);
      return response.data;
    } catch (error) {
      console.error("Profile update error:", error);
      throw error;
    }
  },

  updateProfileImage: async (imageFile) => {
    try {
      const formData = new FormData();
      formData.append("file", imageFile);

      const response = await api.post(
        "/profile/upload-profile-image",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error("Profile image upload error:", error);
      throw error;
    }
  },

  // In your profileService.js file, update the deleteAccount function:
  deleteAccount: async () => {
    try {
      const response = await api.delete('/profile/delete-account');
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      console.error("Delete account error:", error);
      throw error;
    }
  },
};
