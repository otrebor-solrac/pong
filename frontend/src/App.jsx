import React, { useState, useEffect, useCallback } from 'react';
import { Gamepad2, BarChart2 } from 'lucide-react';
import PongCanvas from './components/PongCanvas';
import ControlsPanel from './components/ControlsPanel';
import AnalyticsDashboard from './components/AnalyticsDashboard';

const API_URL = import.meta.env.VITE_API_URL || '';

const initialTelemetry = {
  currentSpeed: 4.5,
  maxSpeed: 4.5,
  speedHistory: [],
  currentRallyTouches: 0,
  ralliesHistory: [],
  maxRally: 0,
  avgRally: 0,
  totalHits: 0,
  p1HitZones: { top: 0, center: 0, bottom: 0 },
  aiHitZones: { top: 0, center: 0, bottom: 0 },
  aiActions: { stay: 0, up: 0, down: 0 },
  inferenceLatency: 0,
  playerScore: 0,
  aiScore: 0
};

export default function App() {
  const [activeTab, setActiveTab] = useState('game'); // 'game' | 'analytics'
  const [player1Mode, setPlayer1Mode] = useState('human');
  const [aiMode, setAiMode] = useState('dqn');
  const [gameRunning, setGameRunning] = useState(false);
  const [ballSpeedMultiplier, setBallSpeedMultiplier] = useState(1.0);
  const [serverStatus, setServerStatus] = useState({ online: false, modelLoaded: false, mode: 'dqn' });
  const [resetTrigger, setResetTrigger] = useState(0);
  const [telemetry, setTelemetry] = useState(initialTelemetry);
  const [recordFailures, setRecordFailures] = useState(true);
  const [failureCount, setFailureCount] = useState(0);

  const handleRestartMatch = useCallback(() => {
    setResetTrigger(prev => prev + 1);
    setGameRunning(false);
    setTelemetry(initialTelemetry);
  }, []);

  const refreshFailureCount = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/failures`);
      if (res.ok) {
        const data = await res.json();
        setFailureCount(data.count);
      }
    } catch (err) {
      // Ignore
    }
  }, []);

  const handleClearFailures = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/failures`, { method: 'DELETE' });
      if (res.ok) {
        setFailureCount(0);
      }
    } catch (err) {
      // Ignore
    }
  }, []);

  const checkServerStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/status`);
      if (res.ok) {
        const data = await res.json();
        setServerStatus({ online: true, modelLoaded: data.model_loaded, mode: data.mode });
      } else {
        setServerStatus({ online: false, modelLoaded: false, mode: 'unknown' });
      }
    } catch (err) {
      setServerStatus({ online: false, modelLoaded: false, mode: 'offline' });
    }
  }, []);

  useEffect(() => {
    checkServerStatus();
    refreshFailureCount();
    const interval = setInterval(() => {
      checkServerStatus();
      refreshFailureCount();
    }, 5000);
    return () => clearInterval(interval);
  }, [checkServerStatus, refreshFailureCount]);

  return (
    <div className="app-container">
      {/* App Header */}
      <header className="app-header">
        <div className="header-brand">
          <h1 className="brand-title">Pong RL</h1>
        </div>

        {/* Top Navigation Tabs */}
        <nav className="header-nav-tabs">
          <button 
            className={`nav-tab-btn ${activeTab === 'game' ? 'active' : ''}`}
            onClick={() => setActiveTab('game')}
            id="tab-match-arena"
          >
            Match
          </button>
          <button 
            className={`nav-tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
            id="tab-analytics"
          >
            Analytics
            {telemetry.currentRallyTouches > 0 && (
              <span className="tab-badge">{telemetry.currentRallyTouches}</span>
            )}
          </button>
        </nav>

        {/* Server Status Indicator */}
        <div className="server-status-pill">
          <span className={`status-dot ${serverStatus.online ? 'online' : 'offline'}`} />
          <span className="status-text">
            {serverStatus.online
              ? (aiMode === 'q_learning' && (telemetry.inferenceLatency === 0 || telemetry.inferenceLatency === undefined)
                  ? 'Rust WASM Active (0 ms)'
                  : (aiMode === 'dqn'
                      ? `FastAPI DQN (${telemetry.inferenceLatency || 0} ms)`
                      : (aiMode === 'q_learning'
                          ? `FastAPI Q-Table (${telemetry.inferenceLatency || 0} ms)`
                          : `Baseline (${aiMode})`)))
              : 'Disconnected'}
          </span>
        </div>
      </header>

      {/* Persistent Views (both mounted to keep 60 FPS physics running uninterrupted in background) */}
      <div className={`tab-view-container ${activeTab === 'game' ? 'view-active' : 'view-hidden'}`}>
        <main className="app-main-grid">
          {/* Left Game Section */}
          <section className="game-section">
            <PongCanvas
              player1Mode={player1Mode}
              aiMode={aiMode}
              apiUrl={API_URL}
              gameRunning={gameRunning}
              setGameRunning={setGameRunning}
              ballSpeedMultiplier={ballSpeedMultiplier}
              resetTrigger={resetTrigger}
              onTelemetryUpdate={setTelemetry}
              recordFailures={recordFailures}
              onFailureRecorded={(count) => setFailureCount(count)}
            />
          </section>

          {/* Right Controls Sidebar */}
          <aside className="sidebar-section">
            <ControlsPanel
              player1Mode={player1Mode}
              onPlayer1ModeChange={setPlayer1Mode}
              aiMode={aiMode}
              onModeChange={setAiMode}
              gameRunning={gameRunning}
              setGameRunning={setGameRunning}
              ballSpeedMultiplier={ballSpeedMultiplier}
              setBallSpeedMultiplier={setBallSpeedMultiplier}
              apiUrl={API_URL}
              onRestartMatch={handleRestartMatch}
              recordFailures={recordFailures}
              onToggleRecordFailures={setRecordFailures}
              failureCount={failureCount}
              onClearFailures={handleClearFailures}
            />
          </aside>
        </main>
      </div>

      {/* Full-Screen Analytics View */}
      <div className={`tab-view-container ${activeTab === 'analytics' ? 'view-active' : 'view-hidden'}`}>
        <AnalyticsDashboard
          telemetry={telemetry}
          player1Mode={player1Mode}
          aiMode={aiMode}
          gameRunning={gameRunning}
          setGameRunning={setGameRunning}
          onResetMetrics={handleRestartMatch}
        />
      </div>
    </div>
  );
}
