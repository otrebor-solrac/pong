import React, { useRef, useEffect, useState, useCallback } from 'react';

const FIELD_WIDTH = 800;
const FIELD_HEIGHT = 500;
const PADDLE_WIDTH = 12;
const PADDLE_HEIGHT = 86;
const BALL_SIZE = 10;
const PLAYER_PADDLE_SPEED = 5.5;
const AI_PADDLE_SPEED = 5.5;
const INITIAL_BALL_SPEED = 4.5;
const PADDLE_FRICTION = 0.35;
const P1_X = 30;
const P2_X = FIELD_WIDTH - 30;

export default function PongCanvas({
  player1Mode = 'human',
  aiMode,
  apiUrl,
  gameRunning,
  setGameRunning,
  ballSpeedMultiplier = 1.0,
  resetTrigger = 0,
  onTelemetryUpdate,
  recordFailures = false,
  onFailureRecorded
}) {
  const canvasRef = useRef(null);
  const wasmEngineRef = useRef(null);

  const loadQTableIntoWasm = useCallback(async () => {
    if (!wasmEngineRef.current) return;
    try {
      const endpoint = apiUrl ? `${apiUrl}/api/q_table` : '/api/q_table';
      const res = await fetch(`${endpoint}?t=${Date.now()}`);
      if (res.ok) {
        const buffer = await res.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        if (typeof wasmEngineRef.current.load_q_table === 'function') {
          const success = wasmEngineRef.current.load_q_table(bytes);
          if (success) {
            console.log('[PongCanvas] Native Rust WASM Q-table loaded successfully (0 ms latency ready).');
          }
        }
      }
    } catch (err) {
      console.warn('[PongCanvas] Could not load Q-table into WASM:', err);
    }
  }, [apiUrl]);

  // Initialize native WebAssembly physics engine & load Q-matrix
  useEffect(() => {
    let active = true;
    import('../wasm/pong_physics.js')
      .then(async (wasm) => {
        if (wasm && wasm.default) {
          await wasm.default();
          if (active && wasm.WasmPongEngine) {
            wasmEngineRef.current = new wasm.WasmPongEngine();
            console.log('[PongCanvas] Native Rust WASM physics engine active.');
            loadQTableIntoWasm();
          }
        }
      })
      .catch((err) => {
        console.warn('[PongCanvas] Could not initialize WASM engine:', err);
      });

    const handleReload = () => {
      loadQTableIntoWasm();
    };
    window.addEventListener('pong:reload_models', handleReload);

    return () => {
      active = false;
      window.removeEventListener('pong:reload_models', handleReload);
    };
  }, [loadQTableIntoWasm]);

  // Re-fetch Q-table when switching to Q-learning mode if not yet loaded
  useEffect(() => {
    if ((aiMode === 'q_learning' || player1Mode === 'q_learning') && wasmEngineRef.current) {
      const isLoaded = typeof wasmEngineRef.current.is_q_table_loaded === 'function'
        ? wasmEngineRef.current.is_q_table_loaded()
        : false;
      if (!isLoaded) {
        loadQTableIntoWasm();
      }
    }
  }, [aiMode, player1Mode, loadQTableIntoWasm]);

  const gameState = useRef({
    ballX: FIELD_WIDTH / 2,
    ballY: FIELD_HEIGHT / 2,
    ballVx: INITIAL_BALL_SPEED,
    ballVy: INITIAL_BALL_SPEED * 0.6 * (Math.random() > 0.5 ? 1 : -1),
    playerY: FIELD_HEIGHT / 2,
    aiY: FIELD_HEIGHT / 2,
    playerScore: 0,
    aiScore: 0,
    keys: { ArrowUp: false, ArrowDown: false, KeyW: false, KeyS: false },
    aiTargetAction: 0,
    p1TargetAction: 0,
    p1RandomAction: 0,
    p2RandomAction: 0,
    lastPredictTime: 0,
    isPredicting: false,
    telemetry: {
      currentRallyTouches: 0,
      totalHits: 0,
      totalPointsCompleted: 0,
      totalTouchesAccumulated: 0,
      maxRally: 0,
      maxSpeed: INITIAL_BALL_SPEED,
      ralliesHistory: [],
      p1HitZones: { top: 0, center: 0, bottom: 0 },
      aiHitZones: { top: 0, center: 0, bottom: 0 },
      aiActions: { stay: 0, up: 0, down: 0 },
      speedHistory: [],
      lastInferenceLatency: 0,
      lastTelemetryPush: 0
    }
  });

  const [scores, setScores] = useState({ player: 0, ai: 0 });

  const resetBall = useCallback((winner) => {
    const state = gameState.current;
    state.ballX = FIELD_WIDTH / 2;
    state.ballY = FIELD_HEIGHT / 2;

    const speed = INITIAL_BALL_SPEED * ballSpeedMultiplier;
    const dirX = winner === 'player' ? 1 : -1;
    const angle = (Math.random() * 0.8 - 0.4);
    state.ballVx = dirX * speed * Math.cos(angle);
    state.ballVy = speed * Math.sin(angle);

    state.lastShotOrigin = {
      ball_x: state.ballX,
      ball_y: state.ballY,
      ball_vx: state.ballVx,
      ball_vy: state.ballVy,
      paddle_x: dirX > 0 ? P1_X : P2_X,
      paddle_y: dirX > 0 ? state.playerY : state.aiY,
      defender_paddle_y: dirX > 0 ? state.aiY : state.playerY,
      p1_paddle_y: state.playerY,
      p2_paddle_y: state.aiY,
      type: 'serve',
      side: dirX > 0 ? 'p1' : 'p2'
    };
  }, [ballSpeedMultiplier]);

  useEffect(() => {
    if (resetTrigger > 0) {
      const state = gameState.current;
      state.ballX = FIELD_WIDTH / 2;
      state.ballY = FIELD_HEIGHT / 2;
      state.playerY = FIELD_HEIGHT / 2;
      state.aiY = FIELD_HEIGHT / 2;
      state.playerScore = 0;
      state.aiScore = 0;
      state.telemetry = {
        currentRallyTouches: 0,
        totalHits: 0,
        totalPointsCompleted: 0,
        totalTouchesAccumulated: 0,
        maxRally: 0,
        maxSpeed: INITIAL_BALL_SPEED * ballSpeedMultiplier,
        ralliesHistory: [],
        p1HitZones: { top: 0, center: 0, bottom: 0 },
        aiHitZones: { top: 0, center: 0, bottom: 0 },
        aiActions: { stay: 0, up: 0, down: 0 },
        speedHistory: [],
        lastInferenceLatency: 0,
        lastTelemetryPush: 0
      };
      setScores({ player: 0, ai: 0 });
      resetBall('player');
    }
  }, [resetTrigger, resetBall, ballSpeedMultiplier]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        if (typeof setGameRunning === 'function') {
          setGameRunning(prev => !prev);
        }
        return;
      }
      if (['ArrowUp', 'ArrowDown', 'KeyW', 'KeyS'].includes(e.code)) {
        e.preventDefault();
        gameState.current.keys[e.code] = true;
      }
    };
    const handleKeyUp = (e) => {
      if (['ArrowUp', 'ArrowDown', 'KeyW', 'KeyS'].includes(e.code)) {
        e.preventDefault();
        gameState.current.keys[e.code] = false;
      }
    };

    window.addEventListener('keydown', handleKeyDown, { passive: false });
    window.addEventListener('keyup', handleKeyUp, { passive: false });
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [setGameRunning]);

  const fetchAiPrediction = useCallback(async () => {
    const state = gameState.current;
    const isWasmQReady = wasmEngineRef.current?.is_q_table_loaded?.();

    // Player 2 (Right Paddle) -> Remote RL API (DQN / DQN No-Blind or fallback if WASM Q-table is not yet ready)
    if (aiMode === 'dqn' || aiMode === 'dqn_noblind' || (aiMode === 'q_learning' && !isWasmQReady)) {
      try {
        const t0 = performance.now();
        const response = await fetch(`${apiUrl}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ball_x: state.ballX,
            ball_y: state.ballY,
            ball_vx: state.ballVx,
            ball_vy: state.ballVy,
            ai_paddle_x: FIELD_WIDTH - 30,
            ai_paddle_y: state.aiY,
            player_paddle_y: state.playerY,
            field_width: FIELD_WIDTH,
            field_height: FIELD_HEIGHT,
            mode: aiMode
          })
        });

        if (response.ok) {
          const data = await response.json();
          state.aiTargetAction = data.action;
          state.telemetry.lastInferenceLatency = Math.max(1, Math.round(performance.now() - t0));
          if (data.action === 0) state.telemetry.aiActions.stay += 1;
          else if (data.action === 1) state.telemetry.aiActions.up += 1;
          else if (data.action === 2) state.telemetry.aiActions.down += 1;
        }
      } catch (err) {
        // Fallback if unreachable
      }
    }

    // Player 1 (Left Paddle) -> Remote RL API (DQN / DQN No-Blind or fallback if WASM Q-table is not yet ready)
    if (player1Mode === 'dqn' || player1Mode === 'dqn_noblind' || (player1Mode === 'q_learning' && !isWasmQReady)) {
      try {
        const response = await fetch(`${apiUrl}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ball_x: state.ballX,
            ball_y: state.ballY,
            ball_vx: state.ballVx,
            ball_vy: state.ballVy,
            ai_paddle_x: 30,
            ai_paddle_y: state.playerY,
            player_paddle_y: state.aiY,
            field_width: FIELD_WIDTH,
            field_height: FIELD_HEIGHT,
            mode: player1Mode
          })
        });

        if (response.ok) {
          const data = await response.json();
          state.p1TargetAction = data.action;
        }
      } catch (err) {
        // Fallback if unreachable
      }
    }
  }, [apiUrl, aiMode, player1Mode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationFrameId;

    const gameLoop = (timestamp) => {
      try {
        const state = gameState.current;

        if (gameRunning) {
        const prevPlayerY = state.playerY;
        const prevAiY = state.aiY;

        // --- 1. Player 1 Movement (Left Paddle - Sky Blue) ---
        if (player1Mode === 'human') {
          if (state.keys.ArrowUp || state.keys.KeyW) {
            state.playerY = Math.max(PADDLE_HEIGHT / 2, state.playerY - PLAYER_PADDLE_SPEED);
          }
          if (state.keys.ArrowDown || state.keys.KeyS) {
            state.playerY = Math.min(FIELD_HEIGHT - PADDLE_HEIGHT / 2, state.playerY + PLAYER_PADDLE_SPEED);
          }
        } else if (player1Mode === 'heuristic') {
          // Heuristic tracking for Player 1 (smooth direct tracking)
          const dy = state.ballY - state.playerY;
          if (dy < -10) {
            state.playerY = Math.max(PADDLE_HEIGHT / 2, state.playerY - AI_PADDLE_SPEED);
          } else if (dy > 10) {
            state.playerY = Math.min(FIELD_HEIGHT - PADDLE_HEIGHT / 2, state.playerY + AI_PADDLE_SPEED);
          }
        } else if (player1Mode === 'random') {
          // Random actions for Player 1
          if (Math.random() < 0.1) {
            state.p1RandomAction = Math.floor(Math.random() * 3);
          }
          if (state.p1RandomAction === 1) {
            state.playerY = Math.max(PADDLE_HEIGHT / 2, state.playerY - AI_PADDLE_SPEED);
          } else if (state.p1RandomAction === 2) {
            state.playerY = Math.min(FIELD_HEIGHT - PADDLE_HEIGHT / 2, state.playerY + AI_PADDLE_SPEED);
          }
        } else if (player1Mode === 'q_learning') {
          // Zero-latency native Rust WebAssembly Q-learning for Player 1
          if (
            wasmEngineRef.current &&
            typeof wasmEngineRef.current.is_q_table_loaded === 'function' &&
            wasmEngineRef.current.is_q_table_loaded() &&
            typeof wasmEngineRef.current.predict_q_action === 'function'
          ) {
            try {
              state.p1TargetAction = wasmEngineRef.current.predict_q_action(
                state.ballX,
                state.ballY,
                state.ballVx,
                state.ballVy,
                30,
                state.playerY
              );
            } catch (err) {
              console.warn('[PongCanvas] WASM predict_q_action p1 error:', err);
            }
          }
          if (state.p1TargetAction === 1) {
            state.playerY = Math.max(PADDLE_HEIGHT / 2, state.playerY - AI_PADDLE_SPEED);
          } else if (state.p1TargetAction === 2) {
            state.playerY = Math.min(FIELD_HEIGHT - PADDLE_HEIGHT / 2, state.playerY + AI_PADDLE_SPEED);
          }
        } else if (player1Mode === 'dqn' || player1Mode === 'dqn_noblind') {
          // Remote DQN action for Player 1
          if (state.p1TargetAction === 1) {
            state.playerY = Math.max(PADDLE_HEIGHT / 2, state.playerY - AI_PADDLE_SPEED);
          } else if (state.p1TargetAction === 2) {
            state.playerY = Math.min(FIELD_HEIGHT - PADDLE_HEIGHT / 2, state.playerY + AI_PADDLE_SPEED);
          }
        }

        // --- 2. Player 2 Movement (Right Paddle - White / AI) ---
        if (aiMode === 'heuristic') {
          // Heuristic tracking for Player 2 (identical smooth local tracking as Player 1)
          const dy = state.ballY - state.aiY;
          if (dy < -10) {
            state.aiY = Math.max(PADDLE_HEIGHT / 2, state.aiY - AI_PADDLE_SPEED);
          } else if (dy > 10) {
            state.aiY = Math.min(FIELD_HEIGHT - PADDLE_HEIGHT / 2, state.aiY + AI_PADDLE_SPEED);
          }
        } else if (aiMode === 'random') {
          // Random actions for Player 2
          if (Math.random() < 0.1) {
            state.p2RandomAction = Math.floor(Math.random() * 3);
          }
          if (state.p2RandomAction === 1) {
            state.aiY = Math.max(PADDLE_HEIGHT / 2, state.aiY - AI_PADDLE_SPEED);
          } else if (state.p2RandomAction === 2) {
            state.aiY = Math.min(FIELD_HEIGHT - PADDLE_HEIGHT / 2, state.aiY + AI_PADDLE_SPEED);
          }
        } else if (aiMode === 'q_learning') {
          // Zero-latency native Rust WebAssembly Q-learning for Player 2 (0.00 ms lag)
          if (
            wasmEngineRef.current &&
            typeof wasmEngineRef.current.is_q_table_loaded === 'function' &&
            wasmEngineRef.current.is_q_table_loaded() &&
            typeof wasmEngineRef.current.predict_q_action === 'function'
          ) {
            try {
              state.aiTargetAction = wasmEngineRef.current.predict_q_action(
                state.ballX,
                state.ballY,
                state.ballVx,
                state.ballVy,
                FIELD_WIDTH - 30,
                state.aiY
              );
              state.telemetry.lastInferenceLatency = 0;
              if (state.aiTargetAction === 0) state.telemetry.aiActions.stay += 1;
              else if (state.aiTargetAction === 1) state.telemetry.aiActions.up += 1;
              else if (state.aiTargetAction === 2) state.telemetry.aiActions.down += 1;
            } catch (err) {
              console.warn('[PongCanvas] WASM predict_q_action p2 error:', err);
            }
          }
          if (state.aiTargetAction === 1) { // UP
            state.aiY = Math.max(PADDLE_HEIGHT / 2, state.aiY - AI_PADDLE_SPEED);
          } else if (state.aiTargetAction === 2) { // DOWN
            state.aiY = Math.min(FIELD_HEIGHT - PADDLE_HEIGHT / 2, state.aiY + AI_PADDLE_SPEED);
          }
        } else if (aiMode === 'dqn' || aiMode === 'dqn_noblind') {
          // Remote DQN inference for Player 2
          if (state.aiTargetAction === 1) { // UP
            state.aiY = Math.max(PADDLE_HEIGHT / 2, state.aiY - AI_PADDLE_SPEED);
          } else if (state.aiTargetAction === 2) { // DOWN
            state.aiY = Math.min(FIELD_HEIGHT - PADDLE_HEIGHT / 2, state.aiY + AI_PADDLE_SPEED);
          }
        }

        const playerVy = state.playerY - prevPlayerY;
        const aiVy = state.aiY - prevAiY;

        // --- 3. Ball Physics with Continuous Sub-Stepping (Anti-Tunneling) ---
        const totalDx = state.ballVx * ballSpeedMultiplier;
        const totalDy = state.ballVy * ballSpeedMultiplier;
        const stepDist = Math.hypot(totalDx, totalDy);
        const subSteps = Math.max(1, Math.ceil(stepDist / 4.0));
        const dtVx = totalDx / subSteps;
        const dtVy = totalDy / subSteps;

        const p1X = 30;
        const p2X = FIELD_WIDTH - 30;
        const p1FrontX = p1X + PADDLE_WIDTH / 2;
        const p2FrontX = p2X - PADDLE_WIDTH / 2;

        // Execute physics: prefer native Rust WASM, fallback to JS
        if (wasmEngineRef.current) {
          try {
            const res = wasmEngineRef.current.step(
              state.ballX,
              state.ballY,
              state.ballVx,
              state.ballVy,
              state.playerY,
              state.aiY,
              playerVy,
              aiVy,
              subSteps,
              ballSpeedMultiplier
            );
            state.ballX = res.ball_x;
            state.ballY = res.ball_y;
            state.ballVx = res.ball_vx;
            state.ballVy = res.ball_vy;

            if (res.hit_p1) {
              const currentSpeed = Math.hypot(res.ball_vx, res.ball_vy);
              state.telemetry.currentRallyTouches += 1;
              state.telemetry.totalHits += 1;
              if (currentSpeed > state.telemetry.currentRallyMaxSpeed) state.telemetry.currentRallyMaxSpeed = currentSpeed;
              if (currentSpeed > state.telemetry.maxSpeed) state.telemetry.maxSpeed = currentSpeed;
              if (res.hit_offset_p1 < -0.33) state.telemetry.p1HitZones.top += 1;
              else if (res.hit_offset_p1 > 0.33) state.telemetry.p1HitZones.bottom += 1;
              else state.telemetry.p1HitZones.center += 1;

              state.lastShotOrigin = {
                ball_x: state.ballX,
                ball_y: state.ballY,
                ball_vx: state.ballVx,
                ball_vy: state.ballVy,
                paddle_x: p1X,
                paddle_y: state.playerY,
                defender_paddle_y: state.aiY,
                p1_paddle_y: state.playerY,
                p2_paddle_y: state.aiY,
                side: 'p1'
              };
            }

            if (res.hit_p2) {
              const currentSpeed = Math.hypot(res.ball_vx, res.ball_vy);
              state.telemetry.currentRallyTouches += 1;
              state.telemetry.totalHits += 1;
              if (currentSpeed > state.telemetry.currentRallyMaxSpeed) state.telemetry.currentRallyMaxSpeed = currentSpeed;
              if (currentSpeed > state.telemetry.maxSpeed) state.telemetry.maxSpeed = currentSpeed;
              if (res.hit_offset_p2 < -0.33) state.telemetry.aiHitZones.top += 1;
              else if (res.hit_offset_p2 > 0.33) state.telemetry.aiHitZones.bottom += 1;
              else state.telemetry.aiHitZones.center += 1;

              state.lastShotOrigin = {
                ball_x: state.ballX,
                ball_y: state.ballY,
                ball_vx: state.ballVx,
                ball_vy: state.ballVy,
                paddle_x: p2X,
                paddle_y: state.aiY,
                defender_paddle_y: state.playerY,
                p1_paddle_y: state.playerY,
                p2_paddle_y: state.aiY,
                side: 'p2'
              };
            }
          } catch (err) {
            console.error('[PongCanvas] WASM physics step error:', err);
          }
        } else {
          // Minimal non-colliding advance while WASM initializes
          state.ballX += totalDx;
          state.ballY += totalDy;
        }


        // Scoring
        if (state.ballX < 0) {
          state.aiScore += 1;
          const touches = state.telemetry.currentRallyTouches;
          if (touches > state.telemetry.maxRally) {
            state.telemetry.maxRally = touches;
          }
          state.telemetry.totalPointsCompleted += 1;
          state.telemetry.totalTouchesAccumulated += touches;
          state.telemetry.ralliesHistory.push({
            id: state.telemetry.totalPointsCompleted,
            touches,
            winner: 'ai',
            score: `${state.playerScore} - ${state.aiScore}`,
            maxSpeed: Math.round(state.telemetry.currentRallyMaxSpeed * 10) / 10
          });
          if (state.telemetry.ralliesHistory.length > 60) {
            state.telemetry.ralliesHistory.shift();
          }

          // Log failure if Player 1 is an RL model
          if (recordFailures && (player1Mode === 'dqn' || player1Mode === 'dqn_noblind' || player1Mode === 'q_learning')) {
            const origin = state.lastShotOrigin || {
              ball_x: FIELD_WIDTH / 2,
              ball_y: FIELD_HEIGHT / 2,
              ball_vx: state.ballVx,
              ball_vy: state.ballVy,
              paddle_x: p2X,
              paddle_y: state.aiY,
              defender_paddle_y: state.playerY,
              p1_paddle_y: state.playerY,
              p2_paddle_y: state.aiY,
              side: 'p2'
            };
            fetch(`${apiUrl}/api/record_failure`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                agent_mode: player1Mode,
                agent_side: 'p1',
                rally_hits: touches,
                peak_speed: state.telemetry.currentRallyMaxSpeed,
                shot_origin: origin,
                miss_impact: {
                  ball_x: state.ballX,
                  ball_y: state.ballY,
                  ball_vx: state.ballVx,
                  ball_vy: state.ballVy,
                  paddle_x: p1X,
                  paddle_y: state.playerY,
                  dy: state.ballY - state.playerY
                }
              })
            })
              .then(res => res.json())
              .then(data => {
                if (data && data.total_failures && onFailureRecorded) {
                  onFailureRecorded(data.total_failures);
                }
              })
              .catch(() => {});
          }

          state.telemetry.currentRallyTouches = 0;
          state.telemetry.currentRallyMaxSpeed = INITIAL_BALL_SPEED * ballSpeedMultiplier;
          setScores({ player: state.playerScore, ai: state.aiScore });
          resetBall('ai');
        } else if (state.ballX > FIELD_WIDTH) {
          state.playerScore += 1;
          const touches = state.telemetry.currentRallyTouches;
          if (touches > state.telemetry.maxRally) {
            state.telemetry.maxRally = touches;
          }
          state.telemetry.totalPointsCompleted += 1;
          state.telemetry.totalTouchesAccumulated += touches;
          state.telemetry.ralliesHistory.push({
            id: state.telemetry.totalPointsCompleted,
            touches,
            winner: 'player1',
            score: `${state.playerScore} - ${state.aiScore}`,
            maxSpeed: Math.round(state.telemetry.currentRallyMaxSpeed * 10) / 10
          });
          if (state.telemetry.ralliesHistory.length > 60) {
            state.telemetry.ralliesHistory.shift();
          }

          // Log failure if Player 2 / AI is an RL model
          if (recordFailures && (aiMode === 'dqn' || aiMode === 'dqn_noblind' || aiMode === 'q_learning')) {
            const origin = state.lastShotOrigin || {
              ball_x: FIELD_WIDTH / 2,
              ball_y: FIELD_HEIGHT / 2,
              ball_vx: state.ballVx,
              ball_vy: state.ballVy,
              paddle_x: p1X,
              paddle_y: state.playerY,
              defender_paddle_y: state.aiY,
              p1_paddle_y: state.playerY,
              p2_paddle_y: state.aiY,
              side: 'p1'
            };
            fetch(`${apiUrl}/api/record_failure`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                agent_mode: aiMode,
                agent_side: 'p2',
                rally_hits: touches,
                peak_speed: state.telemetry.currentRallyMaxSpeed,
                shot_origin: origin,
                miss_impact: {
                  ball_x: state.ballX,
                  ball_y: state.ballY,
                  ball_vx: state.ballVx,
                  ball_vy: state.ballVy,
                  paddle_x: p2X,
                  paddle_y: state.aiY,
                  dy: state.ballY - state.aiY
                }
              })
            })
              .then(res => res.json())
              .then(data => {
                if (data && data.total_failures && onFailureRecorded) {
                  onFailureRecorded(data.total_failures);
                }
              })
              .catch(() => {});
          }

          state.telemetry.currentRallyTouches = 0;
          state.telemetry.currentRallyMaxSpeed = INITIAL_BALL_SPEED * ballSpeedMultiplier;
          setScores({ player: state.playerScore, ai: state.aiScore });
          resetBall('player');
        }

        // Remote API Prediction Polling (Every 16ms / per-frame, only when DQN or un-cached Q-learning is in use)
        const isWasmQReady = wasmEngineRef.current?.is_q_table_loaded?.();
        const needsRemotePrediction =
          (aiMode === 'dqn' || aiMode === 'dqn_noblind' || (aiMode === 'q_learning' && !isWasmQReady)) ||
          (player1Mode === 'dqn' || player1Mode === 'dqn_noblind' || (player1Mode === 'q_learning' && !isWasmQReady));

        if (timestamp - state.lastPredictTime > 16 && !state.isPredicting && needsRemotePrediction) {
          state.isPredicting = true;
          fetchAiPrediction().finally(() => { state.isPredicting = false; });
          state.lastPredictTime = timestamp;
        }

        // Periodic telemetry push (Every 80ms)
        if (timestamp - state.telemetry.lastTelemetryPush > 80) {
          state.telemetry.lastTelemetryPush = timestamp;
          const currentSpeed = Math.hypot(state.ballVx, state.ballVy);
          if (currentSpeed > state.telemetry.maxSpeed) {
            state.telemetry.maxSpeed = currentSpeed;
          }

          const spd = Math.round(currentSpeed * 10) / 10;
          state.telemetry.speedHistory.push(spd);
          if (state.telemetry.speedHistory.length > 30) {
            state.telemetry.speedHistory.shift();
          }

          const avgRally = state.telemetry.totalPointsCompleted > 0
            ? Number((state.telemetry.totalTouchesAccumulated / state.telemetry.totalPointsCompleted).toFixed(1))
            : 0;

          if (onTelemetryUpdate) {
            onTelemetryUpdate({
              currentSpeed: spd,
              maxSpeed: Math.round(state.telemetry.maxSpeed * 10) / 10,
              speedHistory: [...state.telemetry.speedHistory],
              currentRallyTouches: state.telemetry.currentRallyTouches,
              ralliesHistory: [...state.telemetry.ralliesHistory],
              totalPointsCompleted: state.telemetry.totalPointsCompleted,
              maxRally: state.telemetry.maxRally,
              avgRally,
              totalHits: state.telemetry.totalHits,
              p1HitZones: { ...state.telemetry.p1HitZones },
              aiHitZones: { ...state.telemetry.aiHitZones },
              aiActions: { ...state.telemetry.aiActions },
              inferenceLatency: state.telemetry.lastInferenceLatency,
              playerScore: state.playerScore,
              aiScore: state.aiScore
            });
          }
        }
      }

      // Safety guard against NaN
        if (!Number.isFinite(state.ballX)) state.ballX = FIELD_WIDTH / 2;
        if (!Number.isFinite(state.ballY)) state.ballY = FIELD_HEIGHT / 2;
        if (!Number.isFinite(state.playerY)) state.playerY = FIELD_HEIGHT / 2;
        if (!Number.isFinite(state.aiY)) state.aiY = FIELD_HEIGHT / 2;

        // Render
        ctx.clearRect(0, 0, FIELD_WIDTH, FIELD_HEIGHT);

        ctx.fillStyle = '#020617';
        ctx.fillRect(0, 0, FIELD_WIDTH, FIELD_HEIGHT);

        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 8]);
        ctx.beginPath();
        ctx.moveTo(FIELD_WIDTH / 2, 0);
        ctx.lineTo(FIELD_WIDTH / 2, FIELD_HEIGHT);
        ctx.stroke();
        ctx.setLineDash([]);

        // Player 1 Paddle (Left, Sky Blue)
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.roundRect(
          30 - PADDLE_WIDTH / 2,
          state.playerY - PADDLE_HEIGHT / 2,
          PADDLE_WIDTH,
          PADDLE_HEIGHT,
          3
        );
        ctx.fill();

        // Player 2 Paddle (Right, White)
        ctx.fillStyle = '#f8fafc';
        ctx.beginPath();
        ctx.roundRect(
          FIELD_WIDTH - 30 - PADDLE_WIDTH / 2,
          state.aiY - PADDLE_HEIGHT / 2,
          PADDLE_WIDTH,
          PADDLE_HEIGHT,
          3
        );
        ctx.fill();

        // Ball
        ctx.fillStyle = '#f8fafc';
        ctx.beginPath();
        ctx.arc(state.ballX, state.ballY, BALL_SIZE / 2, 0, Math.PI * 2);
        ctx.fill();
      } catch (loopErr) {
        console.error('[PongCanvas] Game loop tick error:', loopErr);
      } finally {
        animationFrameId = requestAnimationFrame(gameLoop);
      }
    };

    animationFrameId = requestAnimationFrame(gameLoop);
    return () => cancelAnimationFrame(animationFrameId);
  }, [gameRunning, ballSpeedMultiplier, fetchAiPrediction, resetBall, player1Mode]);

  return (
    <div className="canvas-wrapper">
      <div className="canvas-header">
        <div className="score-badge player">
          <span className="score-label">P1 ({player1Mode})</span>
          <span className="score-value">{scores.player}</span>
        </div>
        <div className="versus">—</div>
        <div className="score-badge ai">
          <span className="score-label">P2 ({aiMode})</span>
          <span className="score-value">{scores.ai}</span>
        </div>
      </div>

      <div
        className="canvas-container"
        onClick={() => {
          if (!gameRunning && typeof setGameRunning === 'function') {
            setGameRunning(true);
          }
        }}
      >
        <canvas
          ref={canvasRef}
          width={FIELD_WIDTH}
          height={FIELD_HEIGHT}
          className="pong-canvas"
        />
        {!gameRunning && (
          <div className="canvas-pause-overlay">
            <button
              className="btn-start-overlay"
              onClick={(e) => {
                e.stopPropagation();
                if (typeof setGameRunning === 'function') {
                  setGameRunning(true);
                }
              }}
            >
              ▶ Start Match
            </button>
            <span className="overlay-hint">Click or press Space to play</span>
          </div>
        )}
      </div>
    </div>
  );
}
