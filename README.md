# 🚦 TrafficFlow AI: Adaptive Traffic Signal Control Using Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-15%20Passed-brightgreen.svg)](https://pytest.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/koushik0327/AI-project)
[![Course](https://img.shields.io/badge/B.Tech-AI%20Capstone%20Project-purple.svg)]()
  
> **Student:** C. Koushik Kumar | **Reg No:** 24BCE2320  
> **GitHub Repository:** [https://github.com/koushik0327/AI-project](https://github.com/koushik0327/AI-project)  
> A complete, self-contained implementation of Deep Q-Network (DQN) reinforcement learning for dynamic traffic signal control at a 4-way intersection, evaluated against a traditional fixed-time controller using real simulation metrics across 4 traffic regimes.

---

## 📌 Project Abstract

Urban intersections governed by static, pre-timed traffic signals suffer from severe vehicle delays, prolonged queues, and elevated fuel waste due to their inability to adapt to fluctuating traffic densities. This project develops **TrafficFlow AI**, an intelligent closed-loop traffic signal control system driven by a **Deep Q-Network (DQN)** agent implemented directly from first principles in **PyTorch** (without high-level wrappers like Stable-Baselines3). 

The agent observes an exact 15-dimensional state vector (4 approach queues, 4 average wait times, 4 arrival-rate estimates, 2 one-hot phase indicators, and 1 elapsed green-time value) and dynamically selects the optimal traffic light phase. Physical safety constraints (10s minimum green duration, 3s yellow clearance, and 1s all-red clearance) are guaranteed by an integrated safety supervisor. Evaluated on **pre-generated arrival schedules** across independent test seeds (1001–1005) with $\epsilon = 0$, the DQN agent achieves a **55.7% reduction in average vehicle waiting time in medium traffic** and a **60.4% reduction in asymmetric rush-hour traffic**, outperforming the fixed-time baseline while maintaining high throughput.

---

## 🔬 Refined Architectural Specifications

1. **Exact 15-Dimensional State Vector:**
   - Indices `[0:3]`: Normalized queue lengths on N, S, E, W ($Q_i / 30.0$).
   - Indices `[4:7]`: Normalized average wait times on N, S, E, W ($W_i / 60.0\text{s}$).
   - Indices `[8:11]`: Normalized rolling arrival-rate estimates on N, S, E, W ($\hat{\lambda}_i / 0.5\text{ veh/s}$).
   - Indices `[12:13]`: One-hot signal phase indicators ($[1, 0]$ for NS Green, $[0, 1]$ for EW Green, $[0, 0]$ for clearance).
   - Index `[14]`: Normalized elapsed green time in current phase ($t_{\text{elapsed}} / 60.0\text{s}$).

2. **DQN Action Semantics & Safety Invariants:**
   - Action $a_t \in \{0, 1\}$ (0: request NS Green, 1: request EW Green).
   - Minimum green time ($10\text{s}$) strictly prevents rapid signal flickering.
   - Maximum green time ($60\text{s}$) prevents approach starvation.
   - 3-second Yellow and 1-second All-Red clearances ensure intersection conflict zones clear safely.
   - Conflicting directions are mathematically impossible to be green simultaneously.

3. **Explicit Multi-Objective Reward Function:**
   $$R_t = - w_q \sum_{i} Q_i - w_w \sum_{i} W_i + w_t \cdot N_{\text{exited}} - w_s \cdot \mathbb{I}(\text{switch})$$
   - $-0.04 \sum Q_i$: Penalizes queue lengths.
   - $-0.01 \sum W_i$: Penalizes accumulated vehicle idle delays.
   - $+0.30 \cdot N_{\text{exited}}$: Rewards vehicles that have completed crossing and exited.
   - $-0.40 \cdot \mathbb{I}(\text{switch})$: Penalizes phase switches to prevent rapid oscillation.

4. **Rigorous Scientific Fairness Protocol:**
   - **Pre-generated arrival schedules:** Both controllers receive the exact same vehicle arrival sequences.
   - **Disjoint seed sets:** Training on seeds 1–40; testing on independent seeds 1001–1005.
   - **Purely greedy evaluation:** $\epsilon = 0.0$ during benchmarking.
   - Evaluated across all 4 traffic regimes: Low, Medium, Rush-Hour, and Asymmetric.

-

## 📁 Project Directory Structure

```text
trafficflow_ai/
│
├── app.py                      # Main Streamlit web application dashboard
├── train_baseline.py           # Multi-regime training & benchmarking script
├── requirements.txt            # Python dependencies
├── README.md                   # Complete installation, execution, and project documentation
│
├── environment/                # Discrete queue-based traffic simulation package
│   ├── __init__.py
│   ├── vehicle.py              # IDM kinematics, stop line decelerations, 2D coords
│   ├── intersection.py         # 4-way geometry, signal phases, safety constraints, queues
│   ├── traffic_generator.py    # Pre-generated arrival schedules & Poisson processes
│   └── traffic_environment.py  # Exact 15-D state vector, safety actions, reward function
│
├── agent/                      # PyTorch Deep Q-Network reinforcement learning package
│   ├── __init__.py
│   ├── neural_network.py       # Compact PyTorch DQN MLP architecture (15 -> 128 -> 64 -> 2)
│   ├── replay_buffer.py        # Circular experience replay buffer (50,000 capacity)
│   ├── dqn_agent.py            # Bellman updates, target network, epsilon decay, save/load
│   └── trainer.py              # Multi-episode training loop with loss logging
│
├── controllers/                # Signal controllers
│   ├── __init__.py
│   ├── fixed_time_controller.py# Deterministic 30s pre-timed baseline
│   └── adaptive_controller.py  # DRL policy controller (eval_mode=True)
│
├── evaluation/                 # Scientific evaluation and benchmarking
│   ├── __init__.py
│   ├── metrics.py              # Precise traffic engineering metrics calculation
│   ├── evaluator.py            # Pre-generated schedule multi-regime benchmark runner
│   └── comparison.py           # Percentage improvement & comparison table generator
│
├── visualization/              # Visualization engines
│   ├── __init__.py
│   ├── intersection_visualizer.py # Top-down vector SVG visualizer of intersection & cars
│   └── charts.py               # Interactive Plotly figures (rewards, loss, queues)
│
├── models/
│   └── dqn_traffic_model.pt    # Pre-trained PyTorch model weights
│
├── results/
│   ├── multi_regime_results.json # Persisted 4-regime evaluation metrics & timeseries
│   ├── benchmark_results.json    # Cached medium regime telemetry
│   ├── benchmark_results.csv     # Tabular benchmark metrics
│   └── training_history.json     # Reward and loss convergence curves
│
├── docs/
│   ├── ACADEMIC_REPORT.md      # Comprehensive 20-mark university project report
│   └── VIVA_QUESTIONS.md       # 18 in-depth viva examination questions and answers
│
└── tests/                      # Automated unit test suite (pytest)
    ├── __init__.py
    ├── test_environment.py     # Kinematics, 15-D state vector, schedule fairness
    ├── test_agent.py           # Network shapes, replay buffer, weight backprop updates
    ├── test_controllers.py     # Fixed-time cycling and adaptive action outputs
    └── test_metrics.py         # Metric correctness and seed reproducibility
```

   - In Medium traffic, highlight the **55.7% cut in wait time**.
   - In Asymmetric traffic, highlight the **60.4% cut in wait time and +4.2% throughput gain**.
   - Emphasize that in Rush-Hour traffic, max queue was higher for DQN because it prioritized high-density throughput, showing authentic, non-fabricated scientific findings.
6. **Minute 7: Examiner Q&A Defense ([Viva Questions Guide](file:///C:/Users/koush/.gemini/antigravity/scratch/trafficflow_ai/docs/VIVA_QUESTIONS.md))**
   - Answer examiner questions using the 18 model answers covering MDP theory, replay buffers, safety constraints, and real-world deployment.
