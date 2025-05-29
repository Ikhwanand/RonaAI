import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import { ErrorBoundary } from "react-error-boundary";

function ErrorFallback({ error }) {
  return (
    <div role="alert">
      <p>Something went wrong:</p>
      <pre>{error.message}</pre>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <ErrorBoundary FallbackComponent={ErrorFallback}>
    <StrictMode>
      <App />
    </StrictMode>
  </ErrorBoundary>
);
