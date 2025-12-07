import React, { useState } from "react";
import "./App.css";

function App() {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleUpload = async (e) => {
        e.preventDefault();
        if (!file) return;

        setLoading(true);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setResult({ error: err.toString() });
        }

        setLoading(false);
    };

    return (
        <div style={{ fontFamily: "Arial", maxWidth: "700px", margin: "40px auto" }}>
            <h2>Animal Classifier</h2>

            <form onSubmit={handleUpload}>
                <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setFile(e.target.files[0])}
                />
                <button type="submit" disabled={!file || loading} style={{ marginLeft: 10 }}>
                    {loading ? "Classifying..." : "Upload & Classify"}
                </button>
            </form>

            <hr />

            {result && (
                <div style={{ marginTop: "20px" }}>
                    {result.error && <p style={{ color: "red" }}>Error: {result.error}</p>}

                    {result.best_prediction && (
                        <>
                            <div style={{ marginTop: "20px", background: "#e7ffe7", padding: "15px", borderRadius: "8px", border: "1px solid #c3e6cb", color: "#155724" }}>
                                <h3 style={{ marginTop: 0 }}>Final Decision</h3>
                                <p style={{ margin: "5px 0" }}><strong>Model:</strong> {result.best_model}</p>
                                <p style={{ margin: "5px 0" }}><strong>Label:</strong> {result.best_prediction}</p>
                                <p style={{ margin: "5px 0" }}><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
                            </div>

                            <h3>Detailed Model Outputs</h3>
                            <pre
                                style={{
                                    background: "#f8f9fa",
                                    padding: "15px",
                                    borderRadius: "8px",
                                    fontSize: "13px",
                                    border: "1px solid #ddd",
                                    textAlign: "left",
                                    overflowX: "auto"
                                }}
                            >
                                {JSON.stringify(result.all_model_outputs, null, 2)}
                            </pre>
                        </>
                    )}
                </div>
            )}
        </div>
    );
}

export default App;
