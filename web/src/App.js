import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import NewRun from "./pages/NewRun";
import Results from "./pages/Results";
import History from "./pages/History";
import Compare from "./pages/Compare";
import Layout from "./components/Layout";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/run" replace />} />
          <Route path="run" element={<NewRun />} />
          <Route path="results/:runId" element={<Results />} />
          <Route path="history" element={<History />} />
          <Route path="compare" element={<Compare />} />
          <Route path="*" element={<Navigate to="/run" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
