# Skin Doctor App - Frontend

A React-based frontend for the Skin Doctor application, providing an intuitive interface for skin health analysis and tracking.

## ✨ Features

- **User Authentication**
  - Secure login and registration
  - JWT token management
  - Protected routes

- **Profile Management**
  - User profile viewing and editing
  - Profile picture upload
  - Password change functionality

- **Skin Analysis**
  - Image upload for AI analysis
  - Real-time analysis results
  - Historical analysis tracking

- **Skin Journal**
  - Daily skin condition logging
  - Journal entry management
  - Progress visualization

- **Product Management**
  - Skincare product catalog
  - AI recommendations
  - Product effectiveness tracking

## 🛠️ Tech Stack

- **Core**
  - React 19
  - Vite 6
  - React Router 7

- **Styling**
  - Tailwind CSS
  - Custom animations

- **State Management**
  - React Hook Forms
  - Context API

- **API Communication**
  - Axios
  - Custom service layer

- **UI/UX**
  - Framer Motion
  - Custom interactive components (Waves, Iridescence)

## 🚀 Getting Started

### Prerequisites
- Node.js v18+
- npm or yarn

### Installation
1. Navigate to project directory:
   ```bash
   cd d:\skin-doctor-app\project\frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Set VITE_API_URL to your backend URL (default: http://localhost:8000)
   ```

4. Start development server:
   ```bash
   npm run dev
   ```

### Build for Production
```bash
npm run build
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── assets/            # Static assets
│   ├── components/         # Reusable components
│   │   ├── common/        # Generic UI components
│   │   ├── ui/           # Interactive UI elements
│   │   └── *.jsx         # Main components
│   ├── pages/             # Application pages
│   ├── services/          # API service layer
│   ├── utils/             # Utility functions
│   ├── App.jsx           # Main application component
│   └── main.jsx          # Entry point
├── public/                # Public assets
├── vite.config.js        # Vite configuration
└── package.json          # Dependencies and scripts
```

## 🔧 Development

### Available Scripts
- `dev`: Start development server
- `build`: Create production build
- `preview`: Preview production build
- `lint`: Run ESLint

### Coding Standards
- Follow React hooks rules
- Use functional components
- Prefer Tailwind CSS for styling
- TypeScript recommended for new components

## 🤝 Contributing
1. Create a feature branch
2. Follow existing code patterns
3. Write clear commit messages
4. Open a pull request

## 📄 License
MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact
For questions or support: [aldirizaldy977@gmail.com](mailto:aldirizaldy977@gmail.com)
