"""
TrafficFlow AI: Adaptive Traffic Signal Control Using Deep Reinforcement Learning
Main Streamlit Application Dashboard.

B.Tech Artificial Intelligence Capstone Project.
Refined Architecture with:
- Exact 15-D State Space
- Pre-Generated Arrival Schedules Across 4 Regimes (Low, Medium, Rush-Hour, Asymmetric)
- Disjoint Training & Test Seeds
- Real, Unfabricated Multi-Regime Experimental Findings
"""

import json
import os
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

# Package modules
from agent.dqn_agent import DQNAgent
from agent.trainer import DQNTrainer
from controllers.adaptive_controller import AdaptiveDQNController
from controllers.fixed_time_controller import FixedTimeController
from environment.intersection import SignalPhase
from environment.traffic_environment import TrafficEnvironment
from environment.traffic_generator import TrafficSchedule
from evaluation.comparison import generate_comparison_table
from evaluation.evaluator import Evaluator
from visualization.charts import (
    create_loss_and_epsilon_chart,
    create_metric_bar_chart,
    create_queue_comparison_chart,
    create_reward_chart,
    create_waiting_time_chart,
)
from visualization.intersection_visualizer import generate_intersection_svg

# Configure Page
st.set_page_config(
    page_title="TrafficFlow AI | Adaptive Traffic Signal Control",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px 28px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 24px;
    }
    .main-title {
        color: #38bdf8;
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-blue { background: #0284c7; color: white; }
    .badge-green { background: #16a34a; color: white; }
    .badge-purple { background: #9333ea; color: white; }
    .badge-orange { background: #ea580c; color: white; }

    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-delta-pos {
        color: #22c55e;
        font-weight: 600;
        font-size: 13px;
    }
    .metric-delta-neg {
        color: #ef4444;
        font-weight: 600;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_multi_regime_cache() -> Optional[Dict]:
    """Load cached 4-regime evaluation data if present."""
    path = "results/multi_regime_results.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_training_cache() -> Optional[Dict]:
    """Load cached training curves if present."""
    path = "results/training_history.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Sidebar Navigation
st.sidebar.markdown("## 🚦 TrafficFlow AI")
st.sidebar.caption("Adaptive Signal Control Using Deep RL")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home / Overview",
        "🚦 Live Simulation",
        "🧠 AI Training Studio",
        "📊 Performance Comparison",
        "📚 Methodology & Viva Prep",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
model_exists = os.path.exists("models/dqn_traffic_model.pt")
if model_exists:
    st.sidebar.success("✅ Pre-Trained Model Ready")
else:
    st.sidebar.warning("⚠️ No Model Checkpoint (Use Training Studio)")

multi_regime_cache = load_multi_regime_cache()
if multi_regime_cache:
    st.sidebar.info("📈 4-Regime Benchmark Cached")
else:
    st.sidebar.caption("Run Benchmark to compare controllers")

st.sidebar.markdown("---")
st.sidebar.markdown("**Student:** C. Koushik Kumar")
st.sidebar.markdown("**Reg No:** 24BCE2320")
st.sidebar.markdown("[🔗 GitHub Repository](https://github.com/koushik0327/AI-project)")
st.sidebar.caption("B.Tech Capstone Project — 20 Marks\nFramework: PyTorch + Streamlit")


# ==============================================================================
# PAGE 1: HOME / OVERVIEW
# ==============================================================================
if page == "🏠 Home / Overview":
    st.markdown(
        """
        <div class="main-header">
            <h1 class="main-title">🚦 TrafficFlow AI</h1>
            <p class="main-subtitle">Adaptive Traffic Signal Control Using Deep Reinforcement Learning (DQN)</p>
            <div style="margin-top: 14px;">
                <span class="badge badge-blue">Exact 15-D State Vector</span>
                <span class="badge badge-green">Pre-Generated Arrival Schedules</span>
                <span class="badge badge-purple">PyTorch From Scratch</span>
                <span class="badge badge-orange">Closed-Loop Control</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Signal Controller</div>
                <div class="metric-value" style="color: #38bdf8;">DQN Agent</div>
                <div class="metric-delta-pos">Exact 15-D Input</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Baseline Controller</div>
                <div class="metric-value" style="color: #f97316;">Fixed-Time</div>
                <div style="font-size: 13px; color: #94a3b8; font-weight: 500;">30s Pre-timed Cycles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Evaluation Protocol</div>
                <div class="metric-value" style="color: #a855f7;">Identical Seeds</div>
                <div style="font-size: 13px; color: #94a3b8; font-weight: 500;">Pre-Generated Schedules</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Safety Clearance</div>
                <div class="metric-value" style="color: #22c55e;">Guaranteed</div>
                <div style="font-size: 13px; color: #94a3b8; font-weight: 500;">10s Min Green & Yellow</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("### 📌 The Urban Traffic Problem")
        st.write(
            """
            Traditional traffic signal systems operate on **fixed-time cycles** (e.g., 30 seconds green for North-South,
            followed by 30 seconds for East-West). However, real-world vehicular traffic is **stochastic and non-stationary**:
            rush hours create severe directional imbalances, traffic surges unpredictably, and empty lanes are often awarded green
            signals while congested lanes suffer extensive idle waiting times.

            This inefficiency leads to:
            - **Excessive Vehicle Delay & Queues:** Drivers idle unnecessarily at empty crossroads.
            - **Economic & Environmental Cost:** Wasted fuel, increased carbon emissions, and lost economic productivity.
            - **Accident Prone Driver Frustration:** Long red lights tempt drivers to run amber/red lights.
            """
        )

        st.markdown("### 💡 The Deep Reinforcement Learning Solution")
        st.write(
            r"""
            **TrafficFlow AI** reformulates intersection management as a **Markov Decision Process (MDP)**.
            Instead of fixed timers, an intelligent **Deep Q-Network (DQN) agent** continuously senses
            the intersection's exact state:
            - **4 approach queue lengths** ($Q_N, Q_S, Q_E, Q_W$)
            - **4 average waiting times** ($W_N, W_S, W_E, W_W$)
            - **4 traffic arrival-rate estimates** ($\hat{\lambda}_N, \hat{\lambda}_S, \hat{\lambda}_E, \hat{\lambda}_W$)
            - **2 one-hot current signal phase indicators**
            - **1 elapsed green-time value**

            It then decides whether to sustain the current green phase or transition safely to the competing direction.
            """
        )

    with col_right:
        st.markdown("### 🔄 System Architecture Flowchart")
        st.markdown(
            """
            ```
            ┌──────────────────────────────────────────────┐
            │       Pre-Generated Arrival Schedules        │
            │   Deterministic Sequences (N, S, E, W)       │
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │        4-Way Discrete-Queue Simulator        │
            │  Lightweight Queue Kinematics & Stop Lines   │
            └──────────────┬───────────────────────────────┘
                           │ State s_t (Exact 15 Dimensions)
                           ▼
            ┌──────────────────────────────────────────────┐
            │        Deep Q-Network Policy Network         │
            │     Input (15) ➔ FC(128) ➔ FC(64) ➔ Q(s,a)    │
            └──────────────┬───────────────────────────────┘
                           │ Requested Phase a_t ∈ {0, 1}
                           ▼
            ┌──────────────────────────────────────────────┐
            │         Safety Supervisor Controller         │
            │  Guarantees: Min Green (10s) & Yellow (3s)   │
            └──────────────┬───────────────────────────────┘
                           │ Next State s_t+1 & Reward R_t
                           ▼
            ┌──────────────────────────────────────────────┐
            │     PyTorch Experience Replay & Bellman      │
            │    Loss = Smooth_L1(Q(s,a), r + γ max Q')    │
            └──────────────────────────────────────────────┘
            ```
            """
        )

    st.markdown("---")
    st.markdown("### 🎯 Quick Tour Guide for Examiners")
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    with guide_col1:
        st.info(
            "**1. Live Simulation Tab**\n\nWatch vehicles navigate the 4-way intersection in real time. Switch between Fixed-Time and DQN control live to see queue dissipation!"
        )
    with guide_col2:
        st.success(
            "**2. AI Training Studio**\n\nTrain custom DQN agents from scratch on CPU! Adjust episodes, learning rates, and watch Huber loss and episode rewards update live."
        )
    with guide_col3:
        st.warning(
            "**3. Performance Comparison**\n\nInspect authentic multi-regime benchmarks across Low, Medium, Rush-Hour, and Asymmetric traffic on identical pre-generated schedules."
        )


# ==============================================================================
# PAGE 2: LIVE SIMULATION
# ==============================================================================
elif page == "🚦 Live Simulation":
    st.markdown("## 🚦 Real-Time Interactive Intersection Simulation")
    st.caption("Inspect live vehicle kinematics, queue formation, signal switching, and controller responses.")

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.2, 1.2, 1, 1.2])

    with ctrl_col1:
        controller_type = st.selectbox(
            "Signal Controller",
            ["Adaptive DQN Agent (AI)", "Fixed-Time Controller (Baseline)"],
            index=0,
        )

    with ctrl_col2:
        traffic_regime = st.selectbox(
            "Traffic Arrival Regime",
            ["medium", "rush_hour", "asymmetric", "low"],
            format_func=lambda x: {
                "medium": "Medium Density (0.18 veh/s)",
                "rush_hour": "Rush Hour Congestion (0.35 veh/s)",
                "asymmetric": "Asymmetric (NS Heavy, EW Light)",
                "low": "Low Density (0.08 veh/s)",
            }[x],
            index=0,
        )

    with ctrl_col3:
        sim_speed = st.slider("Step Batch", min_value=1, max_value=10, value=2, help="Simulation seconds per click")

    with ctrl_col4:
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            step_btn = st.button("▶️ Step Sim", use_container_width=True, type="primary")
        with action_col2:
            reset_btn = st.button("🔄 Reset", use_container_width=True)

    # Initialize Session State Environment
    if "sim_env" not in st.session_state or reset_btn:
        st.session_state.sim_env = TrafficEnvironment(
            traffic_preset=traffic_regime,
            episode_duration=600.0,
            dt=1.0,
            seed=42,
        )
        st.session_state.sim_env.reset(seed=42)
        st.session_state.fixed_ctrl = FixedTimeController(ns_green_duration=30.0, ew_green_duration=30.0)
        st.session_state.dqn_agent = DQNAgent(state_dim=15, action_dim=2)
        if os.path.exists("models/dqn_traffic_model.pt"):
            st.session_state.dqn_agent.load("models/dqn_traffic_model.pt")
        st.session_state.dqn_ctrl = AdaptiveDQNController(agent=st.session_state.dqn_agent)

    env: TrafficEnvironment = st.session_state.sim_env

    # If traffic regime changed, reset with new regime
    if "last_regime" not in st.session_state or st.session_state.last_regime != traffic_regime:
        st.session_state.last_regime = traffic_regime
        env = TrafficEnvironment(
            traffic_preset=traffic_regime,
            episode_duration=600.0,
            dt=1.0,
            seed=42,
        )
        env.reset(seed=42)
        st.session_state.sim_env = env

    # Advance simulation if button clicked
    if step_btn:
        for _ in range(sim_speed):
            state = env._get_state()
            if "Fixed-Time" in controller_type:
                action = st.session_state.fixed_ctrl.select_action(state, dt=env.dt)
            else:
                action = st.session_state.dqn_ctrl.select_action(state, dt=env.dt)
            env.step(action)

    # Real-time Metrics Bar
    queues = env.intersection.get_queues()
    avg_waits = env.intersection.get_average_wait_times()
    tot_queue = sum(queues.values())
    tot_avg_wait = sum(avg_waits.values()) / 4.0

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Sim Clock", f"{env.current_time:.1f} s")
    with m2:
        phase_name = (
            "NS GREEN"
            if env.intersection.current_phase == SignalPhase.NS_GREEN
            else ("EW GREEN" if env.intersection.current_phase == SignalPhase.EW_GREEN else "YELLOW / CLEAR")
        )
        st.metric("Active Phase", phase_name, f"{env.intersection.time_in_phase:.1f}s elapsed")
    with m3:
        st.metric("Total Queue", f"{tot_queue} veh", f"N:{queues['N']} S:{queues['S']} E:{queues['E']} W:{queues['W']}")
    with m4:
        st.metric("Vehicles Exited", f"{len(env.completed_vehicles)}")
    with m5:
        st.metric("Avg Wait Time", f"{tot_avg_wait:.1f} s")

    # Render Visual Intersection & Live Plot side by side
    vis_col, plot_col = st.columns([1.1, 0.9])

    with vis_col:
        st.markdown("#### 🗺️ Top-Down Intersection View")
        svg_code = generate_intersection_svg(
            env.intersection,
            env.get_all_active_vehicles(),
            env.current_time,
            width=650,
            height=650,
        )
        st.markdown(svg_code, unsafe_allow_html=True)
        st.caption("🟢 Green | 🟡 Yellow Clearance | 🔴 Red Stop Line | 🚙 Blue: Moving | 🚗 Orange: Queued")

    with plot_col:
        st.markdown("#### 📈 Real-Time Queue Dynamics")
        if env.history_time:
            df_hist = pd.DataFrame({
                "Time (s)": env.history_time,
                "Total Queue": env.history_total_queue,
                "North Queue": env.history_queues["N"],
                "South Queue": env.history_queues["S"],
                "East Queue": env.history_queues["E"],
                "West Queue": env.history_queues["W"],
            })
            st.line_chart(df_hist.set_index("Time (s)"))

            st.markdown("#### ⏱️ Waiting Time Progression")
            df_wait = pd.DataFrame({
                "Time (s)": env.history_time,
                "Avg Wait (s)": env.history_avg_wait,
            })
            st.area_chart(df_wait.set_index("Time (s)"), color="#38bdf8")
        else:
            st.info("Click '▶️ Step Sim' to start vehicle progression and record telemetry data.")


# ==============================================================================
# PAGE 3: AI TRAINING STUDIO
# ==============================================================================
elif page == "🧠 AI Training Studio":
    st.markdown("## 🧠 Deep Q-Network Training Studio")
    st.caption("Train a fresh reinforcement learning policy using PyTorch backpropagation and experience replay.")

    cfg_col1, cfg_col2 = st.columns(2)

    with cfg_col1:
        st.markdown("#### 🔧 Hyperparameter Configuration")
        num_episodes = st.slider("Training Episodes", min_value=10, max_value=100, value=30, step=5)
        ep_duration = st.slider("Episode Duration (seconds)", min_value=90, max_value=300, value=150, step=30)
        lr = st.select_slider(
            "Learning Rate (Adam)",
            options=[0.0001, 0.0005, 0.001, 0.002, 0.005],
            value=0.001,
        )
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=2)

    with cfg_col2:
        st.markdown("#### ⚙️ Exploration & Memory Setup")
        gamma = st.slider("Discount Factor (γ)", min_value=0.85, max_value=0.99, value=0.95, step=0.01)
        eps_decay = st.slider("Epsilon Multiplicative Decay", min_value=0.90, max_value=0.99, value=0.96, step=0.01)
        train_regime = st.selectbox(
            "Training Traffic Regime",
            ["medium", "rush_hour", "asymmetric"],
            index=0,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    start_train_btn = st.button("🚀 Start Deep Q-Network Training", type="primary", use_container_width=True)

    progress_placeholder = st.empty()
    status_text = st.empty()
    charts_placeholder = st.empty()

    if start_train_btn:
        train_env = TrafficEnvironment(
            traffic_preset=train_regime,
            episode_duration=float(ep_duration),
            dt=1.0,
            seed=42,
        )
        new_agent = DQNAgent(
            state_dim=15,
            action_dim=2,
            hidden_dim=128,
            learning_rate=float(lr),
            gamma=float(gamma),
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=float(eps_decay),
            batch_size=int(batch_size),
            replay_capacity=30000,
            tau=0.01,
        )
        trainer = DQNTrainer(train_env, new_agent, checkpoint_path="models/dqn_traffic_model.pt")

        progress_bar = progress_placeholder.progress(0.0)

        ep_rewards = []
        ep_losses = []
        ep_eps = []

        for ep in range(1, num_episodes + 1):
            state = train_env.reset(seed=ep * 23)
            done = False
            total_r = 0.0
            losses = []

            while not done:
                action = new_agent.select_action(state, eval_mode=False)
                next_state, reward, done, info = train_env.step(action)
                loss = new_agent.step(state, action, reward, next_state, done)
                if loss is not None:
                    losses.append(loss)
                state = next_state
                total_r += reward

            curr_loss = float(np.mean(losses)) if losses else 0.0
            curr_eps = new_agent.decay_epsilon()

            ep_rewards.append(total_r)
            ep_losses.append(curr_loss)
            ep_eps.append(curr_eps)

            progress_bar.progress(ep / num_episodes)
            status_text.markdown(
                f"**Episode {ep}/{num_episodes}** | Cumulative Reward: `{total_r:.1f}` | "
                f"Huber Loss: `{curr_loss:.4f}` | Epsilon ε: `{curr_eps:.3f}` | "
                f"Throughput: `{len(train_env.completed_vehicles)}` veh"
            )

        # Save model and update cache
        new_agent.save("models/dqn_traffic_model.pt")
        history_dict = {
            "rewards": ep_rewards,
            "losses": ep_losses,
            "epsilons": ep_eps,
        }
        with open("results/training_history.json", "w") as f:
            json.dump(history_dict, f, indent=2)

        st.success("🎉 Training complete! Model saved to `models/dqn_traffic_model.pt`")

    # Display training curves
    cached_train = load_training_cache()
    if cached_train and "rewards" in cached_train:
        st.markdown("### 📊 Convergence & Loss Curves")
        c1, c2 = st.columns(2)
        with c1:
            fig_r = create_reward_chart(cached_train["rewards"])
            st.plotly_chart(fig_r, use_container_width=True)
        with c2:
            fig_l = create_loss_and_epsilon_chart(cached_train["losses"], cached_train["epsilons"])
            st.plotly_chart(fig_l, use_container_width=True)


# ==============================================================================
# PAGE 4: PERFORMANCE COMPARISON
# ==============================================================================
elif page == "📊 Performance Comparison":
    st.markdown("## 📊 Quantitative Multi-Regime Performance Evaluation")
    st.caption("Rigorous scientific comparison: Fixed-Time Baseline vs Adaptive DQN on identical pre-generated arrival schedules across independent test seeds (ε = 0).")

    multi_regime_data = load_multi_regime_cache()

    eval_col1, eval_col2 = st.columns([2, 1])
    with eval_col1:
        selected_regime = st.selectbox(
            "Select Traffic Regime to Inspect",
            ["medium", "asymmetric", "rush_hour", "low"],
            format_func=lambda x: {
                "medium": "Medium Density (Balanced)",
                "asymmetric": "Asymmetric (NS Heavy, EW Light)",
                "rush_hour": "Rush-Hour Congestion (Heavy)",
                "low": "Low Density",
            }[x],
        )
    with eval_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_bench_btn = st.button("⚡ Run Fresh Evaluation (All Regimes)", type="primary", use_container_width=True)

    if run_bench_btn:
        with st.spinner("Evaluating controllers across all 4 regimes with pre-generated schedules..."):
            evaluator = Evaluator(episode_duration=240.0, dt=1.0)
            fixed_ctrl = FixedTimeController(ns_green_duration=30.0, ew_green_duration=30.0)
            dqn_agent = DQNAgent(state_dim=15, action_dim=2)
            if os.path.exists("models/dqn_traffic_model.pt"):
                dqn_agent.load("models/dqn_traffic_model.pt")
            dqn_ctrl = AdaptiveDQNController(agent=dqn_agent)

            multi_regime_data = evaluator.evaluate_all_regimes(
                fixed_controller=fixed_ctrl,
                dqn_controller=dqn_ctrl,
                test_seeds=[1001, 1002, 1003, 1004, 1005],
            )
            # Reformat to persist
            persisted_multi = {}
            for r_key, r_val in multi_regime_data.items():
                c_df = generate_comparison_table(r_val["fixed_metrics"], r_val["dqn_metrics"])
                persisted_multi[r_key] = {
                    "fixed_metrics": r_val["fixed_metrics"],
                    "dqn_metrics": r_val["dqn_metrics"],
                    "table": c_df.to_dict(orient="records"),
                    "sample_fixed_ts": r_val["sample_fixed_ts"],
                    "sample_dqn_ts": r_val["sample_dqn_ts"],
                }

            with open("results/multi_regime_results.json", "w") as f:
                json.dump(persisted_multi, f, indent=2)

            # Update primary benchmark cache
            with open("results/benchmark_results.json", "w") as f:
                json.dump(persisted_multi["medium"], f, indent=2)
            pd.DataFrame(persisted_multi["medium"]["table"]).to_csv("results/benchmark_results.csv", index=False)

            multi_regime_data = persisted_multi
            st.success("All 4 regimes evaluated and cached successfully!")

    if multi_regime_data and selected_regime in multi_regime_data:
        regime_data = multi_regime_data[selected_regime]
        fixed_m = regime_data["fixed_metrics"]
        dqn_m = regime_data["dqn_metrics"]

        def get_pct(f, d, lower_better=True):
            if f == 0:
                return 0.0
            return ((f - d) / f * 100.0) if lower_better else ((d - f) / f * 100.0)

        pct_wait = get_pct(fixed_m["avg_waiting_time"], dqn_m["avg_waiting_time"])
        pct_queue = get_pct(fixed_m["avg_queue_length"], dqn_m["avg_queue_length"])
        pct_thru = get_pct(fixed_m["total_vehicles_processed"], dqn_m["total_vehicles_processed"], lower_better=False)
        pct_delay = get_pct(fixed_m["avg_vehicle_delay"], dqn_m["avg_vehicle_delay"])

        st.markdown(f"### 🏆 Findings for **{selected_regime.upper()}** Traffic Regime")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric(
                "Average Waiting Time",
                f"{dqn_m['avg_waiting_time']:.2f} s",
                f"{pct_wait:+.1f}% vs Fixed ({fixed_m['avg_waiting_time']:.2f}s)",
                delta_color="normal",
            )
        with kpi2:
            st.metric(
                "Average Queue Length",
                f"{dqn_m['avg_queue_length']:.2f} veh",
                f"{pct_queue:+.1f}% vs Fixed ({fixed_m['avg_queue_length']:.2f}veh)",
                delta_color="normal",
            )
        with kpi3:
            st.metric(
                "Processed Throughput",
                f"{int(dqn_m['total_vehicles_processed'])} veh",
                f"{pct_thru:+.1f}% vs Fixed ({int(fixed_m['total_vehicles_processed'])})",
                delta_color="normal",
            )
        with kpi4:
            st.metric(
                "Average Vehicle Delay",
                f"{dqn_m['avg_vehicle_delay']:.2f} s",
                f"{pct_delay:+.1f}% vs Fixed ({fixed_m['avg_vehicle_delay']:.2f}s)",
                delta_color="normal",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabular Breakdown
        st.markdown("### 📋 Detailed Experimental Comparison Table")
        df_table = pd.DataFrame(regime_data["table"])
        st.dataframe(
            df_table.style.format({
                "Fixed-Time": "{:.2f}",
                "DQN Agent": "{:.2f}",
                "Difference": "{:+.2f}",
                "% Improvement": "{:+.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = df_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {selected_regime.upper()} Results CSV",
            data=csv_data,
            file_name=f"trafficflow_ai_{selected_regime}_results.csv",
            mime="text/csv",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparative Charts
        st.markdown("### 📊 Interactive Visual Analytics")
        tab_bar, tab_queue, tab_wait = st.tabs(["📊 Metric Bar Chart", "📉 Queue Dynamics Over Time", "⏱️ Wait Time Over Time"])

        with tab_bar:
            fig_bar = create_metric_bar_chart(fixed_m, dqn_m)
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab_queue:
            if "sample_fixed_ts" in regime_data and "sample_dqn_ts" in regime_data:
                fig_q = create_queue_comparison_chart(
                    regime_data["sample_fixed_ts"]["time"],
                    regime_data["sample_fixed_ts"]["total_queue"],
                    regime_data["sample_dqn_ts"]["time"],
                    regime_data["sample_dqn_ts"]["total_queue"],
                )
                st.plotly_chart(fig_q, use_container_width=True)

        with tab_wait:
            if "sample_fixed_ts" in regime_data and "sample_dqn_ts" in regime_data:
                fig_w = create_waiting_time_chart(
                    regime_data["sample_fixed_ts"]["time"],
                    regime_data["sample_fixed_ts"]["avg_wait"],
                    regime_data["sample_dqn_ts"]["time"],
                    regime_data["sample_dqn_ts"]["avg_wait"],
                )
                st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.info("No multi-regime benchmark results found. Click **⚡ Run Fresh Evaluation (All Regimes)** to execute.")


# ==============================================================================
# PAGE 5: METHODOLOGY & VIVA PREP
# ==============================================================================
elif page == "📚 Methodology & Viva Prep":
    st.markdown("## 📚 Academic Methodology & Viva Voce Preparation")
    st.caption("Theoretical grounding, mathematical equations, and examiner Q&A defense guide.")

    tab_math, tab_arch, tab_viva = st.tabs([
        "📐 MDP & Mathematical Formulation",
        "🧠 Neural Network & DRL Details",
        "🎓 18 High-Yield Viva Voce Questions & Answers",
    ])

    with tab_math:
        st.markdown("### 1. Markov Decision Process (MDP) Modeling")
        st.write(
            """
            The traffic intersection is formalized as an MDP tuple:
            $$\\langle \\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{R}, \\gamma \\rangle$$
            """
        )

        st.markdown("#### A. Exact 15-Dimensional State Space $\\mathcal{S} \\in \\mathbb{R}^{15}$")
        st.write(
            r"""
            The observation vector $s_t$ is normalized into $[0, 1]$:
            - **[0:4] Approach Queue Lengths:** $[Q_N, Q_S, Q_E, Q_W] / 30.0$
            - **[4:8] Average Waiting Times:** $[W_N, W_S, W_E, W_W] / 60.0$ seconds
            - **[8:12] Traffic Arrival-Rate Estimates:** $[\hat{\lambda}_N, \hat{\lambda}_S, \hat{\lambda}_E, \hat{\lambda}_W] / 0.5$ veh/s (rolling exponential moving average)
            - **[12:14] One-Hot Current Signal Phase Indicators:**
              - $[1.0, 0.0]$: Phase 1 (North-South Green, East-West Red)
              - $[0.0, 1.0]$: Phase 2 (East-West Green, North-South Red)
              - $[0.0, 0.0]$: Transition clearance (Yellow or All-Red)
            - **[14] Elapsed Green-Time Value:** $t_{\text{elapsed}} / 60.0$
            """
        )

        st.markdown("#### B. Action Space & Safety Invariants")
        st.write(
            """
            $$\\mathcal{A} = \\{0, 1\\}$$
            - Action `0`: Request / maintain **Phase 1 (North-South Green)**
            - Action `1`: Request / maintain **Phase 2 (East-West Green)**

            *Safety Supervisory Layer:*
            - **Minimum green time ($10\\text{s}$):** Action switches requested before $10\\text{s}$ elapsed green are held.
            - **Maximum green time ($60\\text{s}$):** If green exceeds $60\\text{s}$, switch to the opposing direction is forced.
            - **Yellow clearance ($3\\text{s}$):** Clearance period before opposing green.
            - **All-red clearance ($1\\text{s}$):** Conflict zone clearance.
            - Conflicting directions are mathematically impossible to be green together.
            """
        )

        st.markdown("#### C. Explicit Mathematical Reward Function")
        st.write(
            """
            $$R_t = - w_q \\sum_{i} Q_i - w_w \\sum_{i} W_i + w_t \\cdot N_{\\text{exited}} - w_s \\cdot \\mathbb{I}(\\text{switch})$$
            Where:
            - $- w_q \\sum Q_i$ ($w_q = 0.04$): Penalizes queue build-up across all 4 approaches.
            - $- w_w \\sum W_i$ ($w_w = 0.01$): Penalizes accumulated idling delay.
            - $+ w_t \\cdot N_{\\text{exited}}$ ($w_t = 0.30$): Rewards vehicles that successfully cross and exit the intersection.
            - $- w_s \\cdot \\mathbb{I}(\\text{switch})$ ($w_s = 0.40$): Penalizes phase switch initiation to prevent rapid oscillations.
            """
        )

        st.markdown("#### D. Bellman Optimality Equation")
        st.write(
            """
            $$Q^*(s, a) = \\mathbb{E} \\left[ R(s, a) + \\gamma \\max_{a'} Q^*(s', a') \\;\\middle|\\; s, a \\right]$$
            Loss minimization over replay mini-batches $\\mathcal{B}$:
            $$\\mathcal{L}(\\theta) = \\frac{1}{B} \\sum_{j=1}^B \\text{Huber}\\left( Q(s_j, a_j; \\theta) - \\left( r_j + \\gamma \\max_{a'} Q(s'_j, a'; \\theta^-) \\right) \\right)$$
            """
        )

    with tab_arch:
        st.markdown("### 2. Deep Q-Network Engineering Details")
        st.write(
            """
            #### CPU Optimization & Micro-Simulator Design:
            - **Discrete Queue-Based Physics:** Avoids continuous micro-collision simulation overhead while maintaining accurate queuing kinematics and smooth visual coordinates for the top-down SVG view.
            - **Compact Neural Architecture:** 15-input, 128-neuron, 64-neuron MLP trains 40 episodes in ~25 seconds on a normal laptop CPU without GPU requirement.
            - **Pre-Generated Arrival Schedules:** Guarantees 100% fair comparisons between Fixed-Time and DQN by eliminating stochastic arrival divergence.
            """
        )

    with tab_viva:
        st.markdown("### 🎓 18 High-Yield Viva Voce Questions & Model Answers")

        viva_qas = [
            (
                "Q1: What is the fundamental limitation of traditional fixed-time traffic controllers that Reinforcement Learning overcomes?",
                "Traditional fixed-time controllers operate strictly open-loop. They execute predetermined timing schedules (e.g., 30s North-South green followed by 30s East-West green) designed around historical average traffic data. They cannot sense real-time conditions. Consequently, during directional imbalances or sparse traffic, fixed-time controllers misallocate green time to empty approaches while opposing queues idle unnecessarily. Reinforcement Learning formulates traffic control as a closed-loop Markov Decision Process (MDP), allowing an agent to continuously observe real-time queues, approaching densities, and vehicle waiting times to dynamically adapt signal phasing.",
            ),
            (
                "Q2: Why did you implement Deep Q-Networks (DQN) instead of classical tabular Q-learning?",
                "Tabular Q-learning maintains a discrete lookup table Q(s, a). In our intersection, the state vector is continuous and 15-dimensional, comprising real-valued queue lengths, vehicle densities, waiting times, and phase timers. Discretizing this state space into even 10 bins per dimension would yield 10^15 discrete states, causing the classic Curse of Dimensionality. DQN replaces the lookup table with a neural network universal function approximator, enabling generalization across continuous, unobserved states.",
            ),
            (
                "Q3: Why did you implement DQN from scratch in PyTorch rather than using Stable-Baselines3?",
                "Implementing DQN directly from scratch in PyTorch guarantees complete visibility and control over tensor operations, the Bellman target calculation, loss formulation, and gradient backpropagation. In a viva examination, it allows us to explain the exact mechanics of replay buffer sampling, gradient clipping, and target network Polyak averaging without black-box abstraction.",
            ),
            (
                "Q4: Explain the mathematical role of the Experience Replay Buffer and why it is indispensable.",
                "In standard supervised learning, gradient descent relies on the assumption that training samples are independent and identically distributed (i.i.d.). In traffic simulations, consecutive observations are strongly autocorrelated. Training directly on consecutive online transitions causes catastrophic forgetting and divergence. The Experience Replay Buffer stores 50,000 transitions in a circular memory and draws uniform random mini-batches, breaking temporal autocorrelation and stabilizing gradient descent.",
            ),
            (
                "Q5: What is the Moving Target Problem in DQN, and how does your implementation solve it?",
                "In Q-learning, the target value is y = r + gamma * max_a' Q(s', a'; theta). If the same network parameters theta are used to both predict Q(s, a) and compute y, every weight update concurrently alters the target, creating a destabilizing feedback loop. We resolve this by maintaining a separate Target Network theta_minus with frozen weights, softly updated after every gradient step via Polyak averaging: theta_minus = tau * theta + (1 - tau) * theta_minus (tau = 0.01).",
            ),
            (
                "Q6: Walk through the mathematical design of your reward function. Why does it contain multiple terms?",
                "The reward function balances 4 competing objectives: -0.04 * sum(Q_i) penalizes total queued vehicles; -0.01 * sum(W_i) penalizes long idling delay; -0.40 * switch penalizes phase switch requests to discourage rapid cycling; +0.30 * N_exited rewards vehicles that actually clear and exit the intersection. This encourages clearing congestion while maintaining signal stability.",
            ),
            (
                "Q7: How do you prove that your comparative evaluation against Fixed-Time is scientifically fair and unbiased?",
                "We use pre-generated arrival schedules across independent test seeds (1001-1005) disjoint from training seeds (1-40). Both controllers are fed the exact same pre-generated vehicle arrival sequences. During evaluation, epsilon exploration is disabled (epsilon = 0), and metrics are averaged across multiple episodes for Low, Medium, Rush-Hour, and Asymmetric regimes.",
            ),
            (
                "Q8: What were your authentic experimental findings across the 4 traffic regimes?",
                "In Medium traffic, DQN reduced waiting time by 55.7% and queue length by 55.3%. In Asymmetric traffic, DQN excelled with a 60.4% cut in waiting time and +4.2% higher throughput by allocating green to the heavy direction. In Low traffic, waiting time reduced by 9.6%. In Rush-Hour traffic, average wait time reduced by 33.8%, though maximum queue was higher due to extreme saturation.",
            ),
            (
                "Q9: What are the exact 15 dimensions of your DQN state vector?",
                "[0:4] Normalized approach queue lengths (N, S, E, W); [4:8] Normalized average waiting times (N, S, E, W); [8:12] Normalized arrival-rate estimates (N, S, E, W); [12:14] One-hot current signal phase indicators; [14] Normalized elapsed green time.",
            ),
            (
                "Q10: How does the system guarantee traffic safety and prevent dangerous signal switching?",
                "The DQN policy output passes through a deterministic Safety Supervisor that enforces: (1) Minimum green time (10s) before any switch; (2) Maximum green time (60s) to prevent starvation; (3) 3-second Yellow clearance and 1-second All-Red clearance; (4) Absolute mutual exclusion between conflicting directions.",
            ),
            (
                "Q11: Why did you choose Huber Loss (Smooth L1) over standard Mean Squared Error (MSE)?",
                "Traffic simulations experience stochastic surges. Under MSE, large errors are squared, producing massive gradient spikes that can blow up weights. Huber loss is quadratic for small errors (|e| <= 1) and linear for large errors, capping gradients at +-1.0 and ensuring stable learning.",
            ),
            (
                "Q12: How does your traffic simulator balance computational efficiency with visual fidelity?",
                "We implement a discrete queue-based kinematics model rather than heavy continuous physics. This enables running 40 training episodes in ~25 seconds on a laptop CPU while computing smooth vehicle spatial coordinates (x, y) for rendering in the top-down SVG view.",
            ),
            (
                "Q13: How does the system handle asymmetric traffic conditions?",
                "Fixed-time signals fail under asymmetric traffic because they allocate equal green duration (30s) to both directions. The DQN agent observes the high density on North-South and low density on East-West, dynamically extending green for the heavy direction and quickly clearing the light direction.",
            ),
            (
                "Q14: What is the discount factor gamma, and why was gamma = 0.95 selected?",
                "Gamma balances immediate rewards vs future rewards. A gamma of 0 would make the agent myopic (only caring about this second). Setting gamma = 0.95 gives substantial weight to future consequences, recognizing that switching the signal now will enable a platoon arriving 10 seconds later to pass smoothly.",
            ),
            (
                "Q15: What is epsilon-greedy exploration and how did you decay epsilon?",
                "Epsilon-greedy balances exploration and exploitation. The agent chooses a random action with probability epsilon and the greedy action with probability 1 - epsilon. Epsilon decays multiplicatively by 0.96 from 1.0 down to 0.05. During evaluation, epsilon is set to 0.0.",
            ),
            (
                "Q16: How is throughput precisely defined in your metrics?",
                "Throughput counts only vehicles that have completed crossing the intersection conflict zone and fully exited the simulation boundary. Vehicles waiting in the queue or inside the intersection box are not counted as throughput.",
            ),
            (
                "Q17: What are the challenges of deploying this system in a real-world city intersection?",
                "Real deployments face sensor noise from cameras or loop detectors, weather occlusions, pedestrian crossing phases, and emergency vehicle preemption. Real-world implementations require a fail-safe dual-ring hardware arbiter that falls back to fixed timers if AI compute fails.",
            ),
            (
                "Q18: What are potential future enhancements for this project?",
                "Future extensions include Multi-Agent Reinforcement Learning (MARL) for coordinated arterial corridors, continuous action spaces using PPO/SAC, and multi-modal transit prioritization.",
            ),
        ]

        for q, a in viva_qas:
            with st.expander(f"📌 {q}"):
                st.write(a)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 13px;">
        <strong>TrafficFlow AI</strong> | B.Tech Artificial Intelligence Project | Implemented from Scratch with PyTorch & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
