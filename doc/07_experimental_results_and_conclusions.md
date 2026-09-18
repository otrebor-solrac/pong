# 7. Experimental results, metrics, and conclusions

## 7.1 Quantitative performance comparison

The standardized evaluation of models was conducted under identical physics conditions (200 test episodes, pure greedy policy $\epsilon = 0$, against the 100% speed deterministic tracking heuristic opponent).

| Metric | Tabular Q-Learning | Rule-Based Tracking Heuristic | DQN (Pretrained & Fine-Tuned) |
| :--- | :---: | :---: | :---: |
| **State representation** | Discrete (5,760 non-homogeneous states) | Analytical target tracking | Continuous $\mathbb{R}^4$ $[dx, dy, vx, vy]$ |
| **Mean rally hits ($\bar{H}$)** | $1.88 \pm 0.4$ | $8.80 \pm 1.2$ | **$12.06 \pm 2.4$** |
| **Goals scored ($\bar{G}$)** | $95.5\%$ | $50.0\%$ | **$51.0\%$** |
| **Maximum recorded rally** | $6$ hits | $21$ hits | **$30$ hits** |
| **Robust quality score** | $5.32$ | $9.67$ | **$10.38$** |
| **Throughput / Evaluation speed** | $\approx 25.0\text{ ep/s}$ | $\approx 14.0\text{ ep/s}$ | $\approx 1.05\text{ ep/s}$ |

---

## 7.2 Analysis of distributional metrics

The standardized evaluation highlights key architectural trade-offs between tabular discretization, analytical heuristics, and deep Q-learning:

1. **Rule-based heuristic baseline (Score: 9.67)**:
   In symmetric self-play (Heuristic vs Heuristic at 100% tracking speed), the goal win rate is exactly balanced at $50.0\%$, with an average rally of $8.80$ hits and a maximum rally of $21$ hits. The heuristic fails primarily when high-velocity edge smashes ($\|\mathbf{v}\| \to 15\text{ px/frame}$) create steep angles exceeding paddle travel time.

2. **Tabular Q-learning (Score: 5.32)**:
   The discrete Q-table quickly converges on offensive edge deflections, winning over $95\%$ of short exchanges when returning from close range. However, state quantization ($5,760$ bins) creates perceptual boundaries that limit long-rally defensive persistence (average $1.88$ hits, max rally $6$ hits).

3. **Deep Q-Network (Score: 10.38)**:
   DQN strictly outperforms the rule-based heuristic across all defensive metrics, sustaining an average of $12.06$ hits per episode and achieving a peak rally of $30$ hits. By processing continuous relative coordinates $\mathbb{R}^4$, the network interpolates smooth positioning adjustments and exploits kinetic momentum transfer to match and defeat the tracking heuristic.

The robust quality score:

$$\text{Score} = \frac{\bar{H} + \text{Med} + P_{25}}{3} - \frac{\text{IQR}}{4} + 5.0 \cdot \bar{G}$$

confirms that DQN provides both higher sustained defense ($\bar{H} = 12.06$) and consistent goal conversion ($\bar{G} = 51.0\%$), surpassing the heuristic ceiling of $9.67$.

---

## 7.3 Key lessons and engineering insights

1. **Simulation parity is foundational**:
   Sharing a single Rust implementation between Python training and client-side WebAssembly rendering eliminated the subtle physical discrepancies that frequently degrade web-deployed reinforcement learning agents.
2. **Continuous collision detection is non-negotiable**:
   At velocities exceeding 10 px/frame, discrete forward Euler integration causes tunneling artifacts. The adaptive substep formula:
   $$N_{\text{substeps}} = \max\left(1, \; \left\lceil \frac{\|\mathbf{v}\|}{4.0} \right\rceil\right)$$
   guaranteed mathematical collision integrity with negligible CPU overhead.
3. **Targeted counterfactual replay accelerates convergence**:
   Rather than waiting hundreds of thousands of random exploration steps for the agent to encounter rare failure states, persisting failed trajectories (`failed_shots.json`) and targeting them with anchor-buffered replay resolved systemic blind spots rapidly.
4. **Anchor gameplay prevents policy collapse**:
   Fine-tuning solely on difficult edge cases induced catastrophic forgetting on standard returns. Pre-populating the replay buffer with full-court anchor gameplay preserved the global value function manifold.

---

## 7.4 Future research and development directions

1. **Temporal memory through recurrent architectures or frame stacking**:
   Extending the state vector from a single Markovian frame $[dx, dy, vx, vy]$ to a stack of $K = 4$ consecutive frames or incorporating an LSTM/GRU layer will allow the policy to implicitly model trajectory curvature, ball acceleration, and wall contact anticipation.
2. **Continuous action control with policy gradients (PPO)**:
   Transitioning from 3 discrete actions (`STAY`, `UP`, `DOWN`) to continuous motor velocity $a \in [-v_{\text{max}}, v_{\text{max}}]$ via Proximal Policy Optimization (PPO) will yield smoother paddle trajectory tracking.
3. **Competitive multi-agent self-play**:
   Training two symmetric policy networks against one another in a zero-sum minimax self-play regime will produce evolving competitive strategies without relying on heuristic bot baselines.
