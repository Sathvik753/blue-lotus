import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/Auth";
import { ProtectedRoute, DeveloperRoute } from "./components/ProtectedRoute";
import Layout from "./components/Layout";

import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Status from "./pages/Status";
import NewRun from "./pages/NewRun";
import Results from "./pages/Results";
import History from "./pages/History";
import Compare from "./pages/Compare";
import Billing from "./pages/Billing";
import Developer from "./pages/Developer";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public */}
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/status" element={<Status />} />

          {/* Authenticated app */}
          <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
            <Route path="/run" element={<NewRun />} />
            <Route path="/results/:runId" element={<Results />} />
            <Route path="/history" element={<History />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/billing" element={<Billing />} />
            <Route path="/developer" element={<DeveloperRoute><Developer /></DeveloperRoute>} />
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
