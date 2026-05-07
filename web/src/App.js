import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { api } from "./utils/api";
import Login from "./pages/Login";

import NewRun from "./pages/NewRun";
import Results from "./pages/Results";
import History from "./pages/History";
import Compare from "./pages/Compare";
import ApiKeys from "./pages/ApiKeys";
import Layout from "./components/Layout";

function PrivateRoute({ children }) {
  return api.isLoggedIn() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<PrivateRoute><Layout /></PrivateRoute>}>
          <Route index element={<Navigate to="/run" replace />} />
          <Route path="run" element={<NewRun />} />
          <Route path="results/:runId" element={<Results />} />
          <Route path="history" element={<History />} />
          <Route path="compare" element={<Compare />} />
          <Route path="keys" element={<ApiKeys />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
