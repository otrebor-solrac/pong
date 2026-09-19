import React from 'react';
import { Play, Pause, RotateCcw } from 'lucide-react';

function AnalyticsDashboard({
  telemetry,
  player1Mode,
  aiMode,
  gameRunning,
  setGameRunning,
  onResetMetrics
}) {
  const {
    currentSpeed = 4.5,
    maxSpeed = 4.5,
    currentRallyTouches = 0,
    ralliesHistory = [],
    totalPointsCompleted = ralliesHistory.length,
    maxRally = 0,
    avgRally = 0,
    playerScore = 0,
    aiScore = 0
  } = telemetry;

  // Render bounded window of recent points for optimal 60 FPS DOM performance
  const chartRallies = ralliesHistory.slice(-30);
  const maxTouchesInHistory = Math.max(6, ...chartRallies.map(r => r.touches || 0));
  const tableRallies = [...ralliesHistory].slice(-50).reverse();

  return (
    <div className="analytics-view">
      {/* Subheader Toolbar */}
      <div className="analytics-toolbar">
        <div className="toolbar-left">
          <div className="match-pill">
            <span className={`status-indicator ${gameRunning ? 'active' : 'idle'}`} />
            <span className="match-state-text">
              {gameRunning ? 'In progress' : 'Paused'}
            </span>
            <span className="match-score-text">
              P1 ({player1Mode}) {playerScore} — {aiScore} ({aiMode})
            </span>
          </div>
        </div>

        <div className="toolbar-actions">
          <button 
            className={`btn-action-primary ${gameRunning ? 'btn-pause' : 'btn-play'}`}
            onClick={() => setGameRunning(!gameRunning)}
          >
            {gameRunning ? <Pause size={14} /> : <Play size={14} />}
            {gameRunning ? 'Pause Match' : 'Resume Match'}
          </button>
          <button 
            className="btn-action-secondary"
            onClick={onResetMetrics}
            title="Reset accumulated match statistics"
          >
            <RotateCcw size={13} />
            Reset Stats
          </button>
        </div>
      </div>

      {/* 4 Clean Metric Cards */}
      <div className="metric-cards-grid">
        <div className="metric-card">
          <div className="metric-card-label">Active rally</div>
          <div className="metric-card-value">{currentRallyTouches}</div>
          <div className="metric-card-note">Hits in ongoing point</div>
        </div>

        <div className="metric-card">
          <div className="metric-card-label">Longest rally</div>
          <div className="metric-card-value">{maxRally}</div>
          <div className="metric-card-note">Record touches before goal</div>
        </div>

        <div className="metric-card">
          <div className="metric-card-label">Average touches</div>
          <div className="metric-card-value">{avgRally.toFixed(1)}</div>
          <div className="metric-card-note">{totalPointsCompleted || ralliesHistory.length} points completed</div>
        </div>

        <div className="metric-card">
          <div className="metric-card-label">Live ball speed</div>
          <div className="metric-card-value">{currentSpeed.toFixed(1)} <span className="metric-unit">px/f</span></div>
          <div className="metric-card-note">Match peak: {maxSpeed.toFixed(1)} px/f</div>
        </div>
      </div>

      {/* Main Chart: Rally Touches History */}
      <div className="panel-card">
        <div className="panel-card-header">
          <div>
            <h3 className="panel-card-title">Touches per goal</h3>
            <p className="panel-card-subtitle">Paddle hits exchanged before each goal (latest {chartRallies.length} points)</p>
          </div>
          <div className="chart-legend">
            <span className="legend-item"><span className="legend-chip p1" /> P1 point</span>
            <span className="legend-item"><span className="legend-chip ai" /> AI point</span>
          </div>
        </div>

        <div className="bars-stage">
          {chartRallies.length > 0 ? (
            <div className="bars-track">
              {chartRallies.map((rally) => {
                const heightPct = Math.min(100, Math.max(14, (rally.touches / maxTouchesInHistory) * 100));
                const isP1 = rally.winner === 'player1';

                return (
                  <div key={rally.id} className="bar-column">
                    <div className="bar-meta-top">
                      <span className="bar-touches-val">{rally.touches}</span>
                      <span className="bar-speed-badge">{rally.maxSpeed || 4.5} px/f</span>
                    </div>

                    <div className="bar-container-slot">
                      <div 
                        className={`bar-solid ${isP1 ? 'p1' : 'ai'}`}
                        style={{ height: `${heightPct}%` }}
                      />
                    </div>

                    <div className="bar-meta-bottom">
                      <span className="bar-point-id">#{rally.id}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="empty-state">
              <p>No goals recorded in this match yet.</p>
              <span>Start the match to track rally touches and ball speed history.</span>
            </div>
          )}
        </div>
      </div>

      {/* Points Table */}
      {tableRallies.length > 0 && (
        <div className="panel-card">
          <div className="panel-card-header">
            <div>
              <h3 className="panel-card-title">Point-by-point log</h3>
              <p className="panel-card-subtitle">Recent chronological points (latest {tableRallies.length} of {totalPointsCompleted || ralliesHistory.length})</p>
            </div>
          </div>

          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Point</th>
                  <th>Score</th>
                  <th>Winner</th>
                  <th>Hits</th>
                  <th>Peak Speed</th>
                </tr>
              </thead>
              <tbody>
                {tableRallies.map((rally) => {
                  const isP1 = rally.winner === 'player1';
                  return (
                    <tr key={rally.id}>
                      <td className="cell-mono">#{rally.id}</td>
                      <td className="cell-mono cell-bold">{rally.score}</td>
                      <td>
                        <span className={`chip-player ${isP1 ? 'p1' : 'ai'}`}>
                          {isP1 ? 'Player 1' : 'AI / P2'}
                        </span>
                      </td>
                      <td className="cell-mono">{rally.touches} hits</td>
                      <td className="cell-mono cell-muted">{rally.maxSpeed || 4.5} px/f</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default React.memo(AnalyticsDashboard);
