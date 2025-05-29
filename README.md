# RonaAI App

A comprehensive web application for skin health analysis and tracking, combining AI-powered diagnostics with personal skincare management.

## ✨ Key Features

- **AI-Powered Skin Analysis**: Upload facial images for instant skin condition assessment, get personalized recommendations, and track skin health progress over time.
- **Personal Skin Journal**: Log daily skin conditions, routines, and lifestyle factors affecting skin health. Visualize progress with charts and metrics.
- **Product Management**: Catalog your skincare products, receive AI-generated product recommendations, and track product effectiveness.
- **Profile Management**: Secure user profiles with personal skin profiles, customizable skin type and concern tracking, and profile image/details management.
- **Skincare Recommendations**: Receive personalized product recommendations.

## 🛠️ Tech Stack

### Frontend

- **Core**: React 19, Vite 6, React Router 7
- **Styling**: Tailwind CSS, Custom animations
- **State Management**: React Hook Forms, Context API
- **API Communication**: Axios, Custom service layer
- **UI/UX**: Framer Motion, Custom interactive components (Waves, Iridescence)
- **Testing**: Vitest, React Testing Library

### Backend

- **Core**: FastAPI (Python 3.12+)
- **Database**: SQLAlchemy ORM (SQLite by default, configurable for PostgreSQL)
- **Authentication**: JWT Authentication, Python-JOSE for security
- **Server**: Uvicorn ASGI server
- **Key Libraries**: Pydantic, Alembic (for migrations - *assuming*), bcrypt, python-dotenv

### AI Components

- **Model**: Gemini (via google-genai)
- **Agent Framework**: Agno

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Node.js v18+
- npm or yarn
- A relational database like PostgreSQL (optional, defaults to SQLite)

### Installation

#### Backend Setup

1.  **Navigate to backend directory**:
    ```bash
    cd backend
    ```
2.  **Create and activate virtual environment**:
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```
3.  **Install dependencies** (using `uv` if available, otherwise `pip`):
    ```bash
    # Recommended: using uv (if installed)
    uv pip install -r requirements.txt 
    # Or, using pip
    # pip install -r requirements.txt 
    # Note: requirements.txt can be generated from pyproject.toml using 'uv pip freeze > requirements.txt' or similar tools.
    ```
4.  **Configure environment**:
    Create a `.env` file in the `backend/app/services` directory (or `backend/app` if preferred, adjust `config.py` accordingly). You can copy from an example if one exists, or create it manually. Key variables often include:
    ```env
    DATABASE_URL="sqlite:///./rona_ai.db" # Or your PostgreSQL connection string
    SECRET_KEY="your_strong_secret_key"
    ALGORITHM="HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES=43200 # e.g., 30 days
    REFRESH_TOKEN_EXPIRE_DAYS=30
    # Add other necessary environment variables like API keys for Gemini, etc.
    ```
    Refer to `backend/app/core/config.py` for settings.

5.  **Initialize Database** (if using Alembic or similar for migrations):
    ```bash
    # alembic upgrade head # (Example if Alembic is used)
    ```
6.  **Run development server**:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The backend will typically be available at `http://localhost:8000`.

#### Frontend Setup

1.  **Navigate to frontend directory**:
    ```bash
    cd frontend
    ```
2.  **Install dependencies**:
    ```bash
    npm install
    # or yarn install
    ```
3.  **Configure environment**:
    Create a `.env` file in the `frontend/src` directory (or `frontend` root, depending on Vite config). Copy `frontend/src/.env.example` if it exists, or create it manually.
    ```env
    VITE_API_URL=http://localhost:8000
    ```
4.  **Start development server**:
    ```bash
    npm run dev
    # or yarn dev
    ```
    The frontend will typically be available at `http://localhost:5173`.

## 📁 Project Structure

```
RonaAI/
├── backend/
│   ├── app/                  # Main application package
│   │   ├── api/              # API routers and endpoints
│   │   ├── core/             # Core components (config, security)
│   │   ├── db/               # Database setup and session management
│   │   ├── models/           # SQLAlchemy ORM models
│   │   ├── schemas/          # Pydantic schemas for data validation
│   │   ├── services/         # Business logic and service layer (e.g., AI agent)
│   │   ├── uploads/          # Directory for user uploads (profile/skin images)
│   │   └── main.py           # FastAPI application entry point
│   ├── .python-version       # Specifies Python version (e.g., 3.12)
│   ├── pyproject.toml        # Project metadata and dependencies (PEP 621)
│   ├── requirements.txt      # Pinned dependencies (optional, can be generated)
│   ├── Dockerfile            # Docker configuration for backend
│   └── README.md             # Backend-specific details (if any)
│
├── frontend/
│   ├── public/               # Static assets (e.g., favicon, ML models)
│   ├── src/
│   │   ├── assets/           # Images, fonts, etc.
│   │   ├── components/       # Reusable React components
│   │   ├── pages/            # Page-level components (routed)
│   │   ├── services/         # API interaction services (e.g., authService.js)
│   │   ├── utils/            # Utility functions
│   │   ├── App.jsx           # Main application component with routing
│   │   ├── main.jsx          # React application entry point
│   │   └── index.css         # Global styles / Tailwind base
│   ├── .env                  # Environment variables for frontend (e.g. VITE_API_URL)
│   ├── package.json          # Project metadata and dependencies
│   ├── vite.config.js        # Vite build tool configuration
│   ├── eslint.config.js      # ESLint configuration
│   ├── Dockerfile            # Docker configuration for frontend
│   └── README.md             # Frontend-specific details
│
├── .dockerignore             # Files to ignore in Docker builds (root level)
├── .gitignore                # Files to ignore by Git (root level)
├── docker-compose.yml        # Docker Compose configuration for multi-container setup
└── README.md                 # This file: Overall project documentation
```

## 🌐 API Documentation

Once the backend server is running, API documentation (Swagger UI) is available at:
`http://localhost:8000/docs`

And ReDoc documentation at:
`http://localhost:8000/redoc`

## 🔧 Development

### Backend

- **Linting/Formatting**: Uses `black` (specified in `pyproject.toml`).
- **Testing**: (Details would go here if test setup is known, e.g., Pytest)

### Frontend

- **Available Scripts** (from `package.json`):
    - `npm run dev`: Start development server.
    - `npm run build`: Create production build.
    - `npm run lint`: Run ESLint.
    - `npm run preview`: Preview production build.
    - `npm test`: Run tests with Vitest.
- **Coding Standards**:
    - Follow React hooks rules.
    - Use functional components.
    - Prefer Tailwind CSS for styling.
    - TypeScript recommended for new components (though current files are `.jsx`).

## 🤝 Contributing

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/YourAmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/YourAmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License. (Assuming, as per previous READMEs. Create a `LICENSE` file if one doesn't exist).

## 📧 Contact

For questions or support, please contact: [aldirizaldy977@gmail.com](mailto:aldirizaldy977@gmail.com)

