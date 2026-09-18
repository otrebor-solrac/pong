# 1. Physical and mathematical foundations

## 1.1 Kinematics in continuous two-dimensional space

The simulation environment is defined on a bounded continuous Euclidean domain $\Omega \subset \mathbb{R}^2$, delimited by width $W = 800.0$ and height $H = 500.0$:

$$\Omega = \left\{ (x, y) \in \mathbb{R}^2 \;\middle|\; 0 \le x \le W, \; 0 \le y \le H \right\}$$

The kinematic state of the ball at any continuous time $t$ is fully described by its planar position vector $\mathbf{p}(t)$ and translational velocity vector $\mathbf{v}(t)$:

$$\mathbf{p}(t) = \begin{bmatrix} x(t) \\ y(t) \end{bmatrix}, \quad \mathbf{v}(t) = \begin{bmatrix} v_x(t) \\ v_y(t) \end{bmatrix}$$

The scalar speed $v(t) = \|\mathbf{v}(t)\|_2 = \sqrt{v_x(t)^2 + v_y(t)^2}$ is analytically bounded by:

$$v_{\text{initial}} \le \|\mathbf{v}(t)\|_2 \le v_{\text{max}}$$

where $v_{\text{initial}} = 4.5\text{ px/frame}$ and $v_{\text{max}} = 15.0\text{ px/frame}$. The ball is modeled as a square rigid body with edge length $L_{\text{ball}} = 10.0$, producing an effective half-extent of $r_{\text{ball}} = \frac{L_{\text{ball}}}{2} = 5.0$.

The two paddles (Player 1 on the left and the autonomous agent on the right) have width $W_{\text{paddle}} = 12.0$ and height $H_{\text{paddle}} = 86.0$. Their horizontal coordinates are fixed:

$$x_{\text{p1}} = 30.0, \quad x_{\text{p2}} = W - 30.0 = 770.0$$

The motion of each paddle is restricted to the vertical axis:

$$\mathbf{p}_{\text{paddle}}(t) = \begin{bmatrix} x_{\text{paddle}} \\ y_{\text{paddle}}(t) \end{bmatrix}, \quad \frac{dy_{\text{paddle}}}{dt} = v_{\text{paddle}}(t)$$

where vertical positions are bounded such that paddles remain strictly inside the arena:

$$\frac{H_{\text{paddle}}}{2} \le y_{\text{paddle}}(t) \le H - \frac{H_{\text{paddle}}}{2}$$

---

## 1.2 The tunneling problem and continuous collision detection (CCD)

In conventional discrete physics simulators employing forward Euler integration with a fixed step $\Delta t = 1.0$:

$$\mathbf{p}(t + \Delta t) = \mathbf{p}(t) + \mathbf{v}(t) \Delta t$$

the phenomenon of **tunneling** arises when the displacement of a body across a single discrete frame exceeds the geometric thickness of an obstacle. Here, the paddle thickness is $W_{\text{paddle}} = 12.0$. When the ball reaches maximum speed $v_{\text{max}} = 15.0\text{ px/frame}$:

$$\|\Delta \mathbf{p}\| = 15.0 > W_{\text{paddle}} = 12.0$$

Under this regime, the ball can be strictly to the left of the paddle collider at step $t$, and strictly to the right at step $t + \Delta t$, completely bypassing the axis-aligned bounding box (AABB) intersection check without triggering a collision.

To prevent tunneling without incurring unnecessary computational overhead during low-speed phases, an **adaptive continuous collision detection (CCD)** algorithm is implemented through dynamic trajectory subdivision. The number of substeps $N_{\text{substeps}}$ is computed as a function of instantaneous ball speed:

$$N_{\text{substeps}} = \max\left(1, \; \left\lceil \frac{\|\mathbf{v}\|}{d_{\text{max}}} \right\rceil\right)$$

where the maximum allowable displacement per substep is fixed at $d_{\text{max}} = 4.0\text{ px}$. Consequently:
- At initial speed $v = 4.5$, $N_{\text{substeps}} = \lceil 4.5 / 4.0 \rceil = 2$.
- At terminal speed $v = 15.0$, $N_{\text{substeps}} = \lceil 15.0 / 4.0 \rceil = 4$.

Each substep integrates over a fractional time scale:

$$\Delta \tau = \frac{1.0}{N_{\text{substeps}}}$$

$$\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \cdot \Delta \tau$$

This guarantees that $\|\mathbf{p}_{k+1} - \mathbf{p}_k\| \le 4.0 < W_{\text{paddle}}$, formally eliminating tunneling.

---

## 1.3 Surface bounce dynamics and angular deflection

### 1.3.1 Elastic wall reflections
The top boundary ($y = 0$) and bottom boundary ($y = H$) act as frictionless, perfectly elastic reflection planes:

$$\text{If } y - r_{\text{ball}} \le 0 \implies y \leftarrow r_{\text{ball}}, \quad v_y \leftarrow -v_y$$

$$\text{If } y + r_{\text{ball}} \ge H \implies y \leftarrow H - r_{\text{ball}}, \quad v_y \leftarrow -v_y$$

### 1.3.2 Paddle impact offset
When a collision between the ball and a paddle is detected, the normalized relative impact offset $\delta$ is computed along the vertical axis:

$$\delta = \text{clamp}\left( \frac{y_{\text{ball}} - y_{\text{paddle}}}{H_{\text{paddle}} / 2}, \; -1.0, \; 1.0 \right)$$

A value of $\delta = 0$ corresponds to a dead-center hit, whereas $\delta = \pm 1$ corresponds to hits on the extreme top or bottom tips of the paddle face.

The base deflection angle $\theta_{\text{base}}$ is directly proportional to this offset:

$$\theta_{\text{base}} = \delta \cdot \theta_{\text{max\_deflection}}$$

where $\theta_{\text{max\_deflection}} = 0.75\text{ rad} \approx 42.97^\circ$.

### 1.3.3 Tangential paddle friction
Real-world paddle interactions impart surface shear. The vertical velocity of the moving paddle transfers tangential momentum to the ball, parameterized by friction coefficient $\mu_{\text{friction}} = 0.35$:

$$v_{y,\text{base}} = v_{\text{current}} \cdot \sin(\theta_{\text{base}})$$

$$I_y = v_{\text{paddle}} \cdot \mu_{\text{friction}}$$

The composite post-impact deflection angle before clamping is derived from the resulting velocity vector components:

$$\theta_{\text{composite}} = \operatorname{atan2}\left( v_{y,\text{base}} + I_y, \; \left| v_{\text{current}} \cdot \cos(\theta_{\text{base}}) \right| \right)$$

---

## 1.4 Kinetic momentum transfer and domain randomization

### 1.4.1 Kinetic energy injection (smash effect)
To capture real racquet physics, an active paddle stroke transfers kinetic energy to the projectile. A static return simply redirects the ball, while an accelerating paddle accelerates the ball upon exit.

The momentum transfer adds a scalar speed boost directly proportional to the paddle absolute velocity:

$$K_{\text{transfer}} = |v_{\text{paddle}}| \cdot \beta_{\text{momentum}}$$

where $\beta_{\text{momentum}} = 0.20$. The raw post-collision speed is computed as:

$$v_{\text{raw}} = \|\mathbf{v}\| \cdot \gamma_{\text{rally}} + K_{\text{transfer}}$$

where $\gamma_{\text{rally}} \sim \mathcal{U}(1.02, 1.08)$ is a stochastic restitution coefficient with mean $\bar{\gamma} \approx 1.05$. The new scalar speed is then clamped within the valid physical envelope:

$$v_{\text{current}} = \text{clamp}(v_{\text{raw}}, \; v_{\text{initial}}, \; v_{\text{max}})$$

### 1.4.2 Bounded domain randomization
To prevent overfitting to deterministic trajectories and encourage policy generalization, an angular disturbance is introduced on every paddle deflection:

$$\epsilon_{\text{jitter}} \sim \mathcal{U}(-0.06, 0.06)\text{ rad}$$

$$\theta_{\text{final}} = \text{clamp}\left(\theta_{\text{composite}} + \epsilon_{\text{jitter}}, \; -\theta_{\text{limit}}, \; \theta_{\text{limit}}\right)$$

where $\theta_{\text{limit}} = 0.95\text{ rad} \approx 54.43^\circ$. The post-collision directional velocity components are assigned as:

$$v_x = \begin{cases} 
+|v_{\text{current}} \cos(\theta_{\text{final}})| & \text{for left paddle (Player 1)} \\
-|v_{\text{current}} \cos(\theta_{\text{final}})| & \text{for right paddle (Player 2 / Agent)}
\end{cases}$$

$$v_y = v_{\text{current}} \sin(\theta_{\text{final}})$$

---

## 1.5 Mathematical formulation of reward shaping

Sparse terminal rewards ($+1$ for scoring, $-1$ for conceding) suffer from severe credit assignment degradation over long continuous rallies. To accelerate convergence and guide the agent toward both stable defensive interception and tactical offensive execution, the step reward is decomposed into distinct analytical components:

$$R(s_t, a_t, s_{t+1}) = R_{\text{terminal}} + R_{\text{contact}}(\delta) + R_{\text{alignment}}(dy, a_t)$$

### 1.5.1 Terminal event reward
Scoring against the opponent is prioritized as the primary objective of the Markov decision process, while conceding incurs a bounded penalty:

$$R_{\text{terminal}} = \begin{cases} 
+18.0 & \text{if goal scored against opponent (ball passes left boundary)} \\
-5.0 & \text{if goal conceded by agent (ball passes right boundary)} \\
0.0 & \text{in active rally}
\end{cases}$$

### 1.5.2 Contact reward and edge bonus
To incentivize reliable paddle interceptions while preserving the incentive for aggressive angled returns, contact reward includes an edge deflection bonus:

$$R_{\text{contact}}(\delta) = \begin{cases}
+3.0 & \text{if collision occurs and } |\delta| > 0.85 \quad (\text{Edge trick-shot bonus}) \\
+2.0 & \text{if collision occurs and } |\delta| \le 0.85 \quad (\text{Standard return}) \\
0.0 & \text{no collision at this step}
\end{cases}$$

where $\delta = \text{clamp}\left( \frac{y_{\text{ball}} - y_{\text{paddle}}}{H_{\text{paddle}} / 2}, \; -1.0, \; 1.0 \right)$.

### 1.5.3 Anticipatory trajectory alignment reward
When the ball moves toward the agent ($v_x > 0$), active tracking shaping is awarded based on vertical distance $|dy| = |y_{\text{ball}} - y_{\text{paddle}}|$:

1. **Global gradient**:
   $$R_{\text{grad}} = 0.12 \cdot \left( 1.0 - \min\left(1.0, \; \frac{|dy|}{H / 2}\right) \right)$$

2. **Zone A: Sweet spot ($|dy| \le 22.0\text{ px}$)**:
   The paddle is already aligned with the incoming trajectory. Remaining stationary to avoid jitter is rewarded:
   $$R_{\text{zoneA}} = 0.08 + \begin{cases} +0.08 & \text{if } a_t = \text{STAY} \\ 0.0 & \text{otherwise} \end{cases}$$

3. **Zone B: Edge hazard ($22.0\text{ px} < |dy| \le 48.0\text{ px}$)**:
   The ball threatens to slip past the outer edge of the paddle ($H_{\text{paddle}}/2 + r_{\text{ball}} = 48\text{ px}$). Actively correcting toward the ball is rewarded, while hesitation is penalized:
   $$R_{\text{zoneB}} = \begin{cases} +0.14 & \text{if moving toward ball } (\operatorname{sgn}(a_t) = \operatorname{sgn}(dy)) \\ -0.06 & \text{if } a_t = \text{STAY} \end{cases}$$

4. **Zone C: Outer pursuit ($|dy| > 48.0\text{ px}$)**:
   $$R_{\text{zoneC}} = \begin{cases} +0.10 & \text{if moving toward ball } (\operatorname{sgn}(a_t) = \operatorname{sgn}(dy)) \\ 0.0 & \text{otherwise} \end{cases}$$

