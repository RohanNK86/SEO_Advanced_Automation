const { useState, useEffect } = React;
const API_BASE = "http://localhost:8001/api";

function App() {
    const [metrics, setMetrics] = useState(null);
    const [pages, setPages] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedPage, setSelectedPage] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);

    useEffect(() => {
        fetchDashboard();
        fetchPages();
    }, []);

    const fetchDashboard = async () => {
        try {
            const res = await fetch(`${API_BASE}/dashboard`);
            const data = await res.json();
            setMetrics(data);
        } catch (e) {
            console.error("Failed to fetch dashboard:", e);
        }
    };

    const fetchPages = async () => {
        try {
            const res = await fetch(`${API_BASE}/pages?per_page=50`);
            const data = await res.json();
            setPages(data);
            setLoading(false);
        } catch (e) {
            console.error("Failed to fetch pages:", e);
            setLoading(false);
        }
    };

    const handleRowClick = async (content_id) => {
        setSelectedPage({ content_id, loading: true });
        setAnalysisResult(null);
        setAnalyzing(true);
        
        try {
            // First get the raw page details
            const detailRes = await fetch(`${API_BASE}/pages/${content_id}`);
            const pageData = await detailRes.json();
            setSelectedPage(pageData);

            // Then trigger the agent analysis (with RAG)
            const analyzeRes = await fetch(`${API_BASE}/analyze/${content_id}?use_rag=true`);
            const analysisData = await analyzeRes.json();
            setAnalysisResult(analysisData);
        } catch (e) {
            console.error("Analysis failed:", e);
        } finally {
            setAnalyzing(false);
        }
    };

    const handleApprove = async (content_id, decision) => {
        try {
            await fetch(`${API_BASE}/approve`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ content_id, decision })
            });
            setSelectedPage(null);
            fetchPages(); // Refresh list to show updated statuses if any
        } catch (e) {
            console.error("Approval failed:", e);
        }
    };

    if (loading || !metrics) {
        return <div className="loader" style={{ marginTop: '20vh', width: '50px', height: '50px' }}></div>;
    }

    return (
        <div>
            <header>
                <h1 className="logo-text">FlyRank Content Intelligence</h1>
                <div style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
                    Status: <span style={{ color: "var(--success)" }}>● Live</span>
                </div>
            </header>

            <main className="main-container">
                {/* Dashboard Metrics */}
                <div className="dashboard-grid">
                    <div className="glass-panel metric-card">
                        <span className="metric-label">Pages Analyzed</span>
                        <span className="metric-value">{metrics.total_pages.toLocaleString()}</span>
                    </div>
                    <div className="glass-panel metric-card">
                        <span className="metric-label">Needs Refresh</span>
                        <span className="metric-value" style={{ color: "var(--orange)" }}>{metrics.needs_refresh.toLocaleString()}</span>
                    </div>
                    <div className="glass-panel metric-card">
                        <span className="metric-label">Avg Refresh Score</span>
                        <span className="metric-value">{metrics.avg_score.toFixed(3)}</span>
                    </div>
                    <div className="glass-panel metric-card">
                        <span className="metric-label">Review Queue</span>
                        <span className="metric-value" style={{ color: "var(--warning)" }}>{metrics.review_queue.toLocaleString()}</span>
                    </div>
                </div>

                {/* Pages Table */}
                <div className="glass-panel">
                    <h2 style={{ marginTop: 0, marginBottom: "20px" }}>Priority Action Queue</h2>
                    <div className="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Content ID</th>
                                    <th>Priority</th>
                                    <th>ML Score</th>
                                    <th>Suggested Action</th>
                                    <th>CTR</th>
                                    <th>Impressions (90d)</th>
                                    <th>Age (Days)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {pages.map(page => (
                                    <tr key={page.content_id} onClick={() => handleRowClick(page.content_id)}>
                                        <td>{page.content_id}</td>
                                        <td>
                                            <span className={`badge ${page.priority.toLowerCase()}`}>
                                                {page.priority}
                                            </span>
                                        </td>
                                        <td>{page.score.toFixed(3)}</td>
                                        <td>{page.action}</td>
                                        <td>{(page.ctr * 100).toFixed(1)}%</td>
                                        <td>{page.impressions_90d.toLocaleString()}</td>
                                        <td>{page.content_age_days}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </main>

            {/* Analysis Modal */}
            {selectedPage && (
                <div className="modal-overlay" onClick={() => setSelectedPage(null)}>
                    <div className="modal-content" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <div>
                                <h3 className="modal-title">Content Analysis: {selectedPage.content_id}</h3>
                                <p className="modal-subtitle">Client: {selectedPage.client_id || 'Unknown'}</p>
                            </div>
                            <button className="close-btn" onClick={() => setSelectedPage(null)}>×</button>
                        </div>
                        
                        <div className="modal-body">
                            {/* Left Column: Metrics */}
                            <div className="detail-section">
                                <h4 className="section-title">📊 Current Metrics</h4>
                                <div className="data-row">
                                    <span className="data-label">Click-Through Rate (CTR)</span>
                                    <span className="data-value">{(selectedPage.ctr * 100 || 0).toFixed(2)}%</span>
                                </div>
                                <div className="data-row">
                                    <span className="data-label">Impressions (90d)</span>
                                    <span className="data-value">{(selectedPage.impressions_90d || 0).toLocaleString()}</span>
                                </div>
                                <div className="data-row">
                                    <span className="data-label">Average Position</span>
                                    <span className="data-value">{(selectedPage.avg_position || 0).toFixed(1)}</span>
                                </div>
                                <div className="data-row">
                                    <span className="data-label">Search Volume</span>
                                    <span className="data-value">{(selectedPage.search_volume || 0).toLocaleString()}</span>
                                </div>
                                <div className="data-row">
                                    <span className="data-label">Content Age</span>
                                    <span className="data-value">{selectedPage.content_age_days || 0} days</span>
                                </div>
                                <div className="data-row">
                                    <span className="data-label">Traffic Trend</span>
                                    <span className="data-value" style={{ color: selectedPage.trend_pct < 0 ? 'var(--danger)' : 'var(--success)' }}>
                                        {selectedPage.trend_pct > 0 ? '+' : ''}{(selectedPage.trend_pct || 0).toFixed(1)}%
                                    </span>
                                </div>
                            </div>

                            {/* Right Column: AI Analysis */}
                            <div className="detail-section">
                                <h4 className="section-title">🤖 AI Agent Analysis</h4>
                                {analyzing ? (
                                    <div className="rag-loading">
                                        <div className="loader"></div>
                                        <p>SEO Agents analyzing content and retrieving knowledge...</p>
                                    </div>
                                ) : analysisResult ? (
                                    <div>
                                        <div className="data-row" style={{ borderBottom: 'none', marginBottom: '8px', paddingBottom: 0 }}>
                                            <span className="data-label">Predicted Refresh Score</span>
                                            <span className="data-value" style={{ fontSize: '1.5rem', color: analysisResult.priority_color }}>
                                                {analysisResult.ml_score.toFixed(3)}
                                            </span>
                                        </div>
                                        <div style={{ marginBottom: '20px' }}>
                                            <span className="badge" style={{ backgroundColor: analysisResult.priority_color, color: '#fff', border: 'none' }}>
                                                {analysisResult.priority_label}
                                            </span>
                                            <span style={{ marginLeft: '10px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                                                Confidence: {analysisResult.confidence}%
                                            </span>
                                        </div>

                                        <h5 style={{ margin: '0 0 10px 0', color: 'var(--text-muted)', fontSize: '0.9rem' }}>Identified Issues:</h5>
                                        <ul style={{ margin: '0 0 20px 0', paddingLeft: '20px', fontSize: '0.9rem' }}>
                                            {analysisResult.reason_labels.map((reason, i) => (
                                                <li key={i}>{reason}</li>
                                            ))}
                                        </ul>

                                        {analysisResult.llm_available ? (
                                            <div>
                                                <h5 style={{ margin: '0 0 10px 0', color: 'var(--text-muted)', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                    ✨ RAG Recommendations
                                                    <span className="badge" style={{ background: 'rgba(59, 130, 246, 0.2)', color: 'var(--accent)', border: 'none', fontSize: '0.65rem' }}>
                                                        {analysisResult.rag_docs_count} sources used
                                                    </span>
                                                </h5>
                                                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '16px', borderRadius: '8px', fontSize: '0.9rem', whiteSpace: 'pre-wrap' }} className="markdown-content">
                                                    {analysisResult.llm_recommendation}
                                                </div>
                                            </div>
                                        ) : (
                                            <div style={{ background: 'rgba(234, 179, 8, 0.1)', border: '1px solid rgba(234, 179, 8, 0.3)', padding: '16px', borderRadius: '8px', fontSize: '0.9rem', color: 'var(--warning)' }}>
                                                <strong>LLM Not Configured</strong><br/>
                                                API key is missing in .env file. RAG knowledge retrieval was simulated but LLM recommendations are disabled. Add API key to enable full AI suggestions.
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div style={{ color: 'var(--danger)' }}>Failed to load analysis.</div>
                                )}
                            </div>
                        </div>

                        <div className="modal-actions">
                            <button className="btn btn-outline" onClick={() => setSelectedPage(null)}>Cancel</button>
                            <button className="btn btn-danger" onClick={() => handleApprove(selectedPage.content_id, 'REJECT')} disabled={analyzing}>Reject Refresh</button>
                            <button className="btn btn-success" onClick={() => handleApprove(selectedPage.content_id, 'APPROVE')} disabled={analyzing}>Approve Action</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
