---
title: "Dual projection"
date: 2026-03-29
description: "Polytope projection, Lagrangian dual, normal cones, and a small interactive μ-slider."
tag: optimization
---
# Dual of polytope projection
### Intro 
I'm currently taking a convex optimization course (EE227BT), and we were talking about the classic problem of projecting a point onto a polytope. TLDR is that the dual problem corresponds to finding a normal direction that maximizes the gap between $$x_0$$ and the set along that direction, and I thought it would be cool to visualize it! (see bottom) 

### Problem 
Given a point $$x_0 \in \mathbb{R}^n$$ and a polytope $$\mathcal{P} = \{ x \in \mathbb{R}^n \mid Ax \le b \}$$ with $$A \in \mathbb{R}^{m \times n}$$, $$b \in \mathbb{R}^m$$, one form writes the (Euclidean) projection as the QP: 

$$\min\limits_{x,\,y} \;\|y\|_2 \quad \text{s.t.} \quad Ax \le b, \quad x - x_0 = y.$$

Here $$y$$ is the displacement from $$x_0$$ to $$x$$, and $$\|y\|_2$$ is the distance to minimize. The Lagrangian is (matching the equality constraint $$x - x_0 - y = 0$$):

$$\mathcal{L}(x,y,\lambda,\mu) = \|y\|_2 + \lambda^\top (Ax - b) + \mu^\top (x - x_0 - y), \quad \lambda \in \mathbb{R}^m_+,\; \mu \in \mathbb{R}^n.$$

We can group terms and write the **dual function** as

$$g(\lambda,\mu) = \underbrace{\inf_{y}\big(\|y\|_2 - \mu^\top y\big)}_{\text{(1)}} + \underbrace{\inf_{x} (A^\top \lambda + \mu)^\top x}_{\text{(2)}} - \mu^\top x_0 - b^\top \lambda.$$

- **(2)** is $$0$$ if $$A^\top \lambda + \mu = 0$$ (i.e. $$\mu = -A^\top \lambda$$), and $$-\infty$$ otherwise.
- **(1)**: $$\inf_{y}(\|y\|_2 - \mu^\top y) = 0$$ if $$\|\mu\|_2 \le 1$$, and $$-\infty$$ if $$\|\mu\|_2 > 1$$. (we can see this by re-writing this expression as a dual norm, here the dual of the 2-norm is the 2-norm. By Cauchy–Schwarz if $$\|\mu\|_2 > 1$$ take $$y \propto \mu$$ to drive the objective to $$-\infty$$; if $$\|\mu\|_2 \le 1$$ the infimum is $$0$$, attained at $$y=0$$.)

Eliminating $$\mu$$ by replacing it with $$A^\top \lambda$$ gives the finite **dual objective**:

$$\begin{aligned}
\text{maximize} \quad &(Ax_0 - b)^\top \lambda \\
\text{subject to} \quad &\lambda \ge 0, \quad \|A^\top \lambda\|_2 \le 1.
\end{aligned}$$

Strong duality holds under Slater (e.g. $$\mathcal{P}$$ has nonempty relative interior), and at optimum, $$x^\star = x_0 + y^\star$$.

### Understanding the dual 
First off, any $$x \in \mathcal{P}$$ has a **normal cone** defined as

$$N_{\mathcal{P}}(x) = \Bigl\{ v : v^\top (z - x) \le 0 \;\text{for all}\; z \in \mathcal{P} \Bigr\},$$

which we can write here as just a non-negative weighted sum of the constraint normals (rows of A): $$A^\top \lambda, \lambda \ge 0.$$

There is a result from convex analysis that $$\text{dist}(x_0, \mathcal{C}) = \max_{\|y\|_* \le 1} (y^\top x_0 - \sup_{x \in \mathcal{C}} y^\top x)$$. In other words, we pick a direction $$y$$ with bounded norm, and measure how far $$x_0$$ sticks out beyond our set in that direction.

Using our set $$\mathcal{C} = \{x : Ax \le b\}$$ we can show that for any feasible $$x$$,

$$
Ax \le b \;\Longrightarrow\; \lambda^\top A x \le \lambda^\top b \;\Longrightarrow\; (A^\top \lambda)^\top x \le b^\top \lambda,
$$

for any $$\lambda \ge 0$$. Therefore,

$$
\sup_{x \in \mathcal{C}} (A^\top \lambda)^\top x \le b^\top \lambda.
$$

And, at optimality (by complementary slackness), this inequality is tight, so we can write

$$
\sup_{x \in \mathcal{C}} (A^\top \lambda)^\top x = b^\top \lambda.
$$

Substituting this into the distance representation gives

$$
\text{dist}(x_0, \mathcal{C})
=
\max_{\|y\|_2 \le 1}
\Big( y^\top x_0 - \sup_{x \in \mathcal{C}} y^\top x \Big)
=
\max_{\lambda \ge 0,\; \|A^\top \lambda\|_2 \le 1}
\Big( (A^\top \lambda)^\top x_0 - b^\top \lambda \Big),
$$

which simplifies to

$$
\max_{\lambda \ge 0,\; \|A^\top \lambda\|_2 \le 1}
(Ax_0 - b)^\top \lambda.
$$

### This is the same dual we just derived!

So our geometric picture of the dual is that for any $$\lambda \ge 0$$, the vector

$$
\mu = A^\top \lambda
$$

lies in the **normal cone** of the polytope (as a nonnegative combination of constraint normals). The constraint $$\|\mu\|_2 \le 1$$ restricts us to unit (or sub-unit) directions.

For such a direction $$\mu$$, the quantity

$$
\mu^\top x_0 - \sup_{x \in \mathcal{C}} \mu^\top x
$$

measures how far $$x_0$$ extends beyond the set in direction $$\mu$$. Geometrically, this corresponds to sliding a hyperplane with normal $$\mu$$ toward the set until it just touches $$\mathcal{C}$$, and then measuring the signed distance from that hyperplane to $$x_0$$.

The dual problem is therefore: **among all valid outward normals (those in the normal cone) with bounded norm, find the one that maximizes how far $$x_0$$ protrudes past the set.**

### Connection to projection

At the optimal solution, let $$x^\star$$ be the projection of $$x_0$$ onto $$\mathcal{C}$$. A standard result in convex analysis says:

$$
x_0 - x^\star \in N_{\mathcal{C}}(x^\star),
$$

so there exists $$\lambda^\star \ge 0$$ such that

$$
x_0 - x^\star = A^\top \lambda^\star.
$$

In other words, the displacement from the projection point to $$x_0$$ is exactly a normal vector to the polytope at $$x^\star$$.

Normalizing this vector (which is enforced by $$\|A^\top \lambda\|_2 \le 1$$ in the dual), we see that the optimal direction chosen by the dual is precisely the direction of the projection. Evaluating the dual objective at this point gives

$$
(Ax_0 - b)^\top \lambda^\star
=
(A^\top \lambda^\star)^\top x_0 - b^\top \lambda^\star
=
\mu^{\star\top} x_0 - \mu^{\star\top} x^\star
=
\|x_0 - x^\star\|_2,
$$

which matches the primal objective.

### Takeaway

So the primal and dual are two views of the same geometry:

- The **primal** finds the closest point in the set to $$x_0$$.
- The **dual** finds the supporting hyperplane (with a valid normal) that maximizes the separation from $$x_0$$.

These two coincide because the direction of maximum separation is exactly the normal direction at the projection point!

### Visualization

Here is a pretty simple interactive figure (2D) that I vibecoded to show what the dual problem is doing. Try dragging the slider to rotate the unit normal **μ**. <span style="color:#81d3f9"><strong>Cyan</strong></span> is the shadow of $$\mathcal{C}$$ on the $$\mu$$-line: the segment from $$\bigl(\min_{x\in\mathcal{C}} \mu^\top x\bigr)\,\mu$$ to $$\bigl(\max_{x\in\mathcal{C}} \mu^\top x\bigr)\,\mu$$. <span style="color:#b482ff"><strong>Purple</strong></span> appears only when $$x_0$$ lies outside the supporting halfspace $$\{x:\mu^\top x \le \max_{z\in\mathcal{C}} \mu^\top z\}$$: it is the part of the $$\mu$$-line from that outer contact to $$(\mu^\top x_0)\,\mu$$ (the “extra” past the polytope in direction $$\mu$$). <span style="color:#ffffff"><strong>White</strong></span> marks $$x_0$$; <span style="color:#78c864"><strong>Green</strong></span> is the Euclidean projection $$x^\star = \Pi_{\mathcal{C}}(x_0)$$ (fixed as you rotate $$\mu$$), with a dashed segment $$x_0 \to x^\star$$. <span style="color:#ff6b6b"><strong>Red</strong></span> markers sit on the $$\mu$$-line at the shadow endpoints and at $$(\mu^\top x_0)\,\mu$$.

<figure class="dual-proj-viz">
  <div class="dual-proj-row">
    <div class="dual-proj-main">
      <canvas id="dual-proj-canvas" width="520" height="400" aria-label="Dual projection: polytope shadow and protrusion along a normal"></canvas>
      <div class="dual-proj-slider">
        <label class="dual-proj-label" for="dual-proj-angle">Rotate μ</label>
        <div class="dual-proj-slider-row">
          <input type="range" id="dual-proj-angle" min="0" max="628" value="90" step="0.001" />
          <button id="dual-proj-snap" type="button">Snap to optimal</button>
        </div>
      </div>
    </div>
    <div class="dual-proj-readout" aria-live="polite"></div>
  </div>
</figure>

<script src="/assets/js/dual-projection-viz.js" defer></script>