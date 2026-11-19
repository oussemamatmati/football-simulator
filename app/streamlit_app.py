"""Interactive possession simulator.

Pick a starting situation (action, outcome, pitch zone) and the model simulates thousands of
possessions from there, showing shot/goal probabilities, where possessions end and example sequences.
Run with:  footballsim app   (or: streamlit run app/streamlit_app.py)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from footballsim import config as C
from footballsim.data import make_token, parse_token
from footballsim.evaluate import load_report
from footballsim.plotting import plot_possession, zone_heatmap
from footballsim.simulate import simulate, summarize, zone_distribution
from footballsim.train import load_checkpoint

st.set_page_config(page_title="Football Possession Simulator", page_icon="⚽", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_checkpoint()


@st.cache_data(show_spinner="Simulating possessions...")
def run_sim(prefix: tuple, n: int, temperature: float, seed: int):
    model, vocab, _ = get_model()
    return simulate(model, vocab, n, prefix=list(prefix), temperature=temperature, seed=seed)


def zone_label(z: int) -> str:
    xi, yi = divmod(z, C.GRID_Y)
    depth = ["own box", "own half", "defensive mid", "attacking mid", "final third", "attacking box"][xi]
    lane = ["left", "centre-left", "centre-right", "right"][yi]
    return f"z{z}: {depth}, {lane}"


st.title("⚽ Football Possession Simulator")
st.caption("A small transformer trained on StatsBomb open data. It generates possessions event by event; "
           "pick a starting situation and see what tends to happen next.")

try:
    model, vocab, ckpt = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Starting situation")
    free = st.checkbox("Let the model choose (no starting state)", value=False)
    action = st.selectbox("First action", ["Pass", "Carry", "Dribble", "Ball Recovery", "Interception"],
                          disabled=free)
    outcome = st.selectbox("Outcome", ["ok", "fail", "na"], index=0 if action == "Pass" else 2, disabled=free,
                           help="Outcome applies to passes (ok/fail); other actions use 'na'.")
    zone = st.selectbox("Pitch zone", list(range(C.N_ZONES)), index=8, format_func=zone_label, disabled=free)
    st.header("Simulation")
    n = st.slider("Possessions to simulate", 500, 10000, 3000, step=500)
    temperature = st.slider("Temperature", 0.5, 1.5, 1.0, 0.05,
                            help="Below 1 = more typical play, above 1 = more varied play.")
    seed = st.number_input("Random seed", 0, 10_000, 0)

prefix = () if free else (make_token(action, outcome, zone),)
if prefix and prefix[0] not in vocab.stoi:
    st.warning(f"`{prefix[0]}` never (or rarely) occurred in the training data. Try another combination.")
    st.stop()

sims = run_sim(prefix, n, temperature, int(seed))
stats = summarize(sims)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Possessions that produce a shot", f"{stats['shot_rate']:.1%}")
c2.metric("Possessions that produce a goal", f"{stats['goal_rate']:.2%}")
c3.metric("Average length (events)", f"{stats['avg_length']:.1f}")
c4.metric("End in the final third", f"{stats['ends_in_final_third']:.1%}")

left, right = st.columns([1, 1])
with left:
    st.subheader("Where possessions end")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = zone_heatmap(ax, zone_distribution(sims, "end"), "Final event zone", cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig)
with right:
    st.subheader("Example possessions")
    k = st.slider("How many to show", 1, 6, 4)
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.4))
    for ax_, toks in zip(axes.ravel(), sims[:k]):
        plot_possession(ax_, toks)
    for ax_ in axes.ravel()[k:]:
        ax_.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Blue dots = events, red stars = shots. Arrows follow the order of events.")

with st.expander("Example sequences as text"):
    for s in sims[:5]:
        st.code(" → ".join(s), language=None)

with st.expander("How good is this model? (held-out evaluation)"):
    try:
        rep = load_report()
        a, b = st.columns(2)
        a.metric("Transformer perplexity", f"{rep['transformer_perplexity']:.2f}",
                 delta=f"{-rep['improvement_pct']:.1f}% vs Markov", delta_color="inverse")
        b.metric("Markov baseline perplexity", f"{rep['markov_perplexity']:.2f}")
        table = pd.DataFrame({"real (held-out)": rep["real"], "simulated": rep["simulated"]}).round(4)
        st.dataframe(table)
    except FileNotFoundError:
        st.info("Run `footballsim evaluate` to generate the held-out evaluation report.")

st.caption("Limitations: ~100 tournament matches; no team, player or score context; coarse 6x4 zones. "
           "Probabilities describe the training distribution, not any specific team.")
