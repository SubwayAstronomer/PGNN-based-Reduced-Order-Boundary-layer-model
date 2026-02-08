
---

# Equation-Free, Physics-Guided Neural Model for Laminar Boundary-Layer Evolution

This repository implements an **equation-free neural operator** that learns the **downstream evolution of laminar boundary-layer velocity profiles** using only data and physics-inspired constraints — **without explicit access to the governing equations** (e.g. Navier–Stokes or the Blasius equation).

The model discovers a **low-dimensional latent representation** of boundary-layer physics and uses it to predict how velocity profiles evolve downstream.

---

## 1. Overview

Laminar boundary layers exhibit strong physical structure:

* no-slip at the wall
* monotonic velocity increase away from the wall
* smooth, viscous diffusion
* self-similar downstream growth

In classical fluid mechanics, these properties are encoded in **similarity solutions** (e.g. Blasius).
In this work, we ask a different question:

> *Can a neural model rediscover this structure purely from data and physical constraints, without being given the equations?*

This repository answers **yes** — for laminar, zero-pressure-gradient boundary layers.

---

## 2. Key Ideas

* **Equation-free learning**
  The model never sees the Navier–Stokes equations or the Blasius ODE.

* **Physics-guided training**
  Physical principles are enforced through **loss functions**, not equations:

  * no-slip condition
  * smoothness (viscous diffusion)
  * monotonicity
  * freestream preservation
  * causal downstream evolution

* **Latent representation**
  High-dimensional velocity profiles are compressed into a **low-dimensional latent space** that captures boundary-layer physics.

* **Operator learning**
  The network learns a mapping
  [
  u(x) ;\rightarrow; u(x+\Delta x)
  ]
  rather than solving a PDE.

---

## 3. Model Architecture

### Encoder–Decoder Structure

```
Velocity Profile u(y)  →  Encoder  →  Latent z
Latent z               →  Decoder  →  Next Profile u(y)
```

* **Encoder**
  Compresses a velocity profile into a latent vector.

* **Latent space**
  Learns physically meaningful coordinates (e.g. boundary-layer thickness, wall shear proxies).

* **Decoder**
  Reconstructs the downstream velocity profile from the latent representation.

---

## 4. Data Generation

### Blasius-Based Synthetic Data (Ground Truth)

Training data is generated from the **exact Blasius similarity solution**, but the model never sees:

* the Blasius equation
* similarity variables
* analytical expressions

Key features of the data generation:

* fixed physical wall-normal grid
* correct similarity scaling
  [
  \eta = y \sqrt{\frac{U_\infty}{\nu x}}
  ]
* exact freestream behavior ((u \to U_\infty))
* exact no-slip condition

This ensures **physically consistent training data** while preserving the equation-free learning paradigm.

---
## 5. Physics-Guided Loss Functions

The total training loss is defined as a weighted combination of a data loss and a physics-based regularization term:

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda\,\mathcal{L}_{\text{physics}}
$$

where:
- $\mathcal{L}_{\text{data}}$ is the mean-squared error between predicted and reference velocity profiles,
- $\mathcal{L}_{\text{physics}}$ encodes known physical constraints of laminar boundary-layer flow,
- $\lambda$ controls the strength of the physics regularization.

### Physics Loss Components

| Loss Term | Physical Meaning |
|---------|------------------|
| **No-slip loss** | Enforces the no-slip boundary condition at the wall: $u(y=0) = 0$ |
| **Smoothness loss** | Penalizes high curvature in the velocity profile, representing viscous diffusion |
| **Monotonicity loss** | Enforces a positive wall-normal velocity gradient near the wall: $\partial u / \partial y > 0$ |
| **Outer-flow loss** | Enforces convergence to the freestream velocity: $u \to U_\infty$ |
| **Causal loss** | Penalizes unphysical jumps between consecutive downstream profiles |

These constraints eliminate non-physical solutions and guide the network toward the correct boundary-layer solution manifold without explicitly enforcing the governing equations.

---

## 6. What the Model Learns

After training, the model:

* Learns a **low-dimensional similarity manifold** for laminar boundary layers
* Discovers **monotonic latent variables** associated with boundary-layer growth
* Accurately predicts downstream evolution via **autoregressive rollout**
* Preserves physical invariants (no-slip, freestream, smoothness)

A sharp drop in training loss corresponds to the moment when the network discovers the correct **internal coordinate system** for boundary-layer evolution.

---

## 7. Generalization Capabilities

### Supported (Works Well)

* Different downstream step sizes
* Different starting locations
* Longer downstream rollouts
* Different Reynolds numbers (within laminar regime)
* New laminar initial profiles on the same similarity manifold

### Not Supported (By Design)

* Pressure gradients (Falkner–Skan)
* Transition or turbulence
* Flow separation
* Arbitrary velocity profiles

The model generalizes **only within the learned laminar physics manifold**, which makes its behavior interpretable and trustworthy.

---

## 8. How to Run

### Install Dependencies

```bash
pip install numpy torch scipy matplotlib
```

### Train the Model

```bash
python main.py
```

This will:

* generate physically consistent laminar boundary-layer data
* train the neural evolution operator
* save the trained model to `outputs/`
* generate diagnostic plots

---

## 9. Outputs

After training, the following plots are generated:

* **Training loss (log scale)**
* **Downstream velocity profiles (prediction vs. truth)**
* **Near-wall velocity behavior**
* **Latent dynamics**
* **Self-similarity collapse**

These plots provide **physical validation**, not just numerical accuracy.

---

## 10. Scientific Interpretation

This work demonstrates that:

> Boundary-layer similarity is not just a mathematical artifact of differential equations, but a structural property that can be rediscovered from data under physical constraints.

The model does not learn equations — it learns **the geometry of the solution space**.

---

## 11. Limitations and Future Work

Planned extensions include:

* pressure-gradient boundary layers (Falkner–Skan)
* conditional models with Reynolds number inputs
* noisy / experimental data
* comparison with turbulence (expected to fail cleanly)

---

## 12. License

This project is intended for **research and educational use**.
Please cite appropriately if used in academic work.

