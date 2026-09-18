# 3. Classical reinforcement learning: tabular Q-learning

## 3.1 Markov Decision Process (MDP) formulation of Pong

The Pong environment is formulated as an infinite-horizon Markov Decision Process (MDP) defined by the 5-tuple:

$$\mathcal{M} = \left( \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \right)$$

where:
- $\mathcal{S}$ is the set of valid environment states.
- $\mathcal{A}$ is the finite set of discrete agent control actions.
- $\mathcal{P}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$ represents the state transition probability distribution dictated by the underlying continuous physics engine and domain randomization.
- $\mathcal{R}(s, a, s')$ is the scalar reward function.
- $\gamma \in [0, 1)$ is the temporal discount factor, configured to $\gamma = 0.99$.

---

## 3.2 State space discretization and feature binning

While the underlying simulation runs in continuous space $\mathbb{R}^4$, tabular reinforcement learning requires a discrete state space $\mathcal{S}_{\text{discrete}} = \{0, 1, \dots, |\mathcal{S}| - 1\}$.

The discretizer (`PongStateDiscretizer`) projects the continuous physics coordinates $(x_{\text{ball}}, y_{\text{ball}}, v_{x,\text{ball}}, v_{y,\text{ball}}, x_{\text{paddle}}, y_{\text{paddle}})$ into a compact linear state index through non-linear spatial and kinematic binning.

### 3.2.1 Relative spatial coordinates
Instead of absolute arena positions, the discretizer computes the relative offset between the ball and the center of the agent paddle:

$$\Delta x = |x_{\text{paddle}} - x_{\text{ball}}|, \quad \Delta y = y_{\text{ball}} - y_{\text{paddle}}$$

These values are clamped to operational limits $\Delta x_{\text{max}} = 740.0$ and $\Delta y_{\text{max}} = 457.0$ and partitioned into uniform discrete bins:

$$b_{\Delta x} = \min\left( n_{\text{dx}} - 1, \; \left\lfloor \frac{\Delta x}{\Delta x_{\text{max}}} \cdot n_{\text{dx}} \right\rfloor \right), \quad n_{\text{dx}} = 20$$

$$b_{\Delta y} = \min\left( n_{\text{dy}} - 1, \; \left\lfloor \frac{\Delta y + \Delta y_{\text{max}}}{2 \Delta y_{\text{max}}} \cdot n_{\text{dy}} \right\rfloor \right), \quad n_{\text{dy}} = 24$$

### 3.2.2 Directional and velocity tier binning
1. **Approach indicator ($b_{v_x}$)**: Binary flag distinguishing whether the ball is moving toward the agent paddle or receding away:
   $$b_{v_x} = \begin{cases} 0 & \text{if approaching} \\ 1 & \text{if receding} \end{cases}$$
2. **Vertical direction ($b_{v_y}$)**: Binary sign of vertical ball motion:
   $$b_{v_y} = \begin{cases} 0 & \text{if } v_y \ge 0\text{ (downward)} \\ 1 & \text{if } v_y < 0\text{ (upward)} \end{cases}$$
3. **Speed magnitude tiers ($b_{\text{speed}}$)**: To account for non-linear ball acceleration during extended rallies, scalar speed $v = \sqrt{v_x^2 + v_y^2}$ is categorized into 3 dynamic tiers:
   $$b_{\text{speed}} = \begin{cases}
   0 & \text{if } v \le 7.0\text{ px/frame (low speed)} \\
   1 & \text{if } 7.0 < v \le 11.0\text{ px/frame (medium speed)} \\
   2 & \text{if } v > 11.0\text{ px/frame (high speed)}
   \end{cases}$$

### 3.2.3 Linear state index mapping
The multi-dimensional coordinate tuple $(b_{\Delta x}, b_{\Delta y}, b_{v_x}, b_{v_y}, b_{\text{speed}})$ is flattened into a single linear integer identifier:

$$\text{state\_id} = b_{\Delta x} \cdot M_{\Delta x} + b_{\Delta y} \cdot M_{\Delta y} + b_{v_x} \cdot M_{v_x} + b_{v_y} \cdot M_{v_y} + b_{\text{speed}}$$

where the stride multipliers are:

$$M_{v_y} = 3, \quad M_{v_x} = 2 \times 3 = 6, \quad M_{\Delta y} = 2 \times 6 = 12, \quad M_{\Delta x} = 24 \times 12 = 288$$

The cardinality of the discrete state space is:

$$|\mathcal{S}| = 20 \times 24 \times 2 \times 2 \times 3 = 5,760\text{ states}$$

---

## 3.3 Discrete action space

The agent selects actions from a discrete set of 3 motor commands:

$$\mathcal{A} = \{0, 1, 2\}$$

where:
- $a = 0$ (`ACTION_STAY`): Paddle remains stationary ($v_{\text{paddle}} = 0$).
- $a = 1$ (`ACTION_UP`): Paddle translates upward by $v_{\text{paddle}} = -v_{\text{paddle\_speed}}$.
- $a = 2$ (`ACTION_DOWN`): Paddle translates downward by $v_{\text{paddle}} = +v_{\text{paddle\_speed}}$.

The agent paddle speed is calibrated to $v_{\text{paddle\_speed}} = 10.0\text{ px/frame}$.

---

## 3.4 Bellman optimality equation and update rule

The tabular agent estimates the optimal action-value function $Q^*(s, a)$, representing the expected cumulative discounted return starting from state $s$, taking action $a$, and following the greedy policy thereafter:

$$Q^*(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q^*(S_{t+1}, a') \;\middle|\; S_t = s, A_t = a \right]$$

### 3.4.1 Temporal Difference (TD) learning
The tabular update rule adjusts the estimated value $Q(s_t, a_t)$ toward the one-step bootstrap target at each simulation step:

$$\text{TD Target} = \begin{cases}
r_{t+1} & \text{if episode terminates at } t+1 \\
r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') & \text{otherwise}
\end{cases}$$

$$\delta_t = \text{TD Target} - Q(s_t, a_t)$$

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha_{\text{eff}} \cdot \delta_t$$

### 3.4.2 Residual error boosting
To accelerate convergence during rare transition events (e.g., unexpected goal concessions or sharp edge bounces), the base learning rate $\alpha = 0.15$ is dynamically scaled by the magnitude of the temporal difference error:

$$\alpha_{\text{eff}} = \alpha \cdot \left( 1.0 + 0.5 \cdot \min(2.0, |\delta_t|) \right)$$

This mechanism functions analogously to gradient boosting, assigning larger gradient steps to transitions with high prediction errors.

### 3.4.3 Backward failure credit assignment
When an episode terminates with a conceded goal ($r \le -4.0$ or unforced miss), relying solely on single-step TD updates propagates the penalty backward at a rate of only one state per episode. 

To overcome this latency, a retrospective trajectory trace of recent state-action pairs $(s_{\tau}, a_{\tau})$ (up to 12 steps prior to the goal) receives an immediate discounted penalty:

$$G_{\tau} \leftarrow r_{\text{terminal}} \cdot (0.85)^{t - \tau}$$

$$Q(s_{\tau}, a_{\tau}) \leftarrow Q(s_{\tau}, a_{\tau}) + 0.5 \alpha \cdot \left( G_{\tau} - Q(s_{\tau}, a_{\tau}) \right)$$

This backward propagation rapidly reinforces defensive corrections along the failure trajectory.

### 3.4.4 Anti-jitter smoothness penalty
To prevent high-frequency oscillations (alternating between `UP` and `DOWN` at consecutive steps), an action-switching cost is deducted from the step reward:

$$R_{\text{effective}} = R(s, a, s') - c_{\text{jitter}} \cdot \mathbb{I}(a_t \neq a_{t-1})$$

with $c_{\text{jitter}} = 0.05$, yielding visibly smoother paddle tracking.

---

## 3.5 Exploration versus exploitation: epsilon-greedy schedule

Action selection follows an $\epsilon$-greedy policy:

$$\pi(a \mid s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
\frac{\epsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}$$

The exploration parameter $\epsilon$ decays exponentially across training episodes:

$$\epsilon_{k+1} = \max\left( \epsilon_{\text{min}}, \; \epsilon_k \cdot \lambda_{\text{decay}} \right)$$

Typical schedule parameters:
- $\epsilon_{\text{start}} = 1.0$ (or $0.20$ during warm-restart on existing matrices).
- $\epsilon_{\text{min}} = 0.01$.
- $\lambda_{\text{decay}} = 0.9992$.

---

## 3.6 Curriculum learning, evaluation protocol, and checkpointing

To prevent policy degradation from exploratory perturbations, the tabular trainer follows a rigorous evaluation and curriculum regime:

### 3.6.1 Curriculum learning against heuristic opponent
When trained against a dynamic paddle, the opponent speed ratio $\rho_{\text{opp}} \in [0.45, 1.00]$ is dynamically scaled with episode index $k \in [1, K]$:

$$\rho_{\text{opp}}(k) = 0.45 + 0.55 \cdot \left( \frac{k - 1}{K - 1} \right)$$

This allows the agent to first master low-speed returns before facing maximum-speed edge deflections.

### 3.6.2 Standardized environmental evaluation
To decouple evaluation scores from the progressive curriculum difficulty:
1. Every report interval, training is paused and exploration is disabled ($\epsilon = 0$).
2. The environment locks the opponent speed ratio to full difficulty ($\rho_{\text{opp}} = 1.00$) for the duration of the evaluation trials ($N = 40$ or $N = 200$).
3. Once completed, the original training difficulty ratio is restored.

### 3.6.3 Robust distributional quality metric
Performance is evaluated through a quartile-based robust scoring function that penalizes erratic play and rewards defensive consistency and goal-scoring capability:

$$\text{Score} = \frac{\bar{H} + \text{Med}(H) + P_{25}(H)}{3} - \frac{\text{IQR}(H)}{4} + 5.0 \cdot \bar{G}$$

where:
- $\bar{H}$ is the empirical mean rally hits.
- $\text{Med}(H)$ is the median hits count.
- $P_{25}(H)$ represents the 25th percentile (defensive skill floor).
- $\text{IQR}(H) = P_{75}(H) - P_{25}(H)$ captures rally length variance.
- $\bar{G}$ represents average goals scored per episode (active in heuristic mode).

### 3.6.4 Baseline protection and immediate checkpointing
When pre-trained matrices are supplied (`--pretrained`), the trainer establishes a baseline score prior to optimization. A checkpoint is committed immediately to disk (`models/q_table.npy`) whenever a periodic evaluation strictly beats the current historical record:

$$\text{Score}_{\text{eval}} > \text{Score}_{\text{best}}$$

---

## 3.7 Convergence properties and fundamental limitations

While the tabular method converges reliably to a baseline defensive policy capable of sustaining rallies against simple wall reflections (averaging 15 to 25 hits per episode), it encounters insurmountable architectural limitations in high-speed competitive play:

1. **Information loss through discretization**:
   With $n_{\text{dy}} = 24$ bins spanning $2 \times 457.0 = 914.0\text{ px}$, each vertical bin covers approximately $38.08\text{ px}$. Because the paddle height is $86.0\text{ px}$, small variations in arrival altitude fall within identical bins, preventing the agent from performing precise edge deflections.
2. **Absence of spatial generalization**:
   The Q-table treats every discrete state $s_i$ as an isolated scalar memory address. Updating $Q(s_i, a)$ yields zero generalization to adjacent spatial states $s_{i+1}$, requiring extensive sampling across all 5,760 cells.
3. **Dimensionality explosion under trajectory memory**:
   Addressing rebound prediction requires conditioning on multiple past positions (e.g., ball acceleration or multi-bounce projections). Adding even two trajectory history bins would expand the table from 5,760 to over 500,000 states, making tabular sample complexity intractable.

These limitations motivate the transition to deep function approximation via Deep Q-Networks (DQN).
