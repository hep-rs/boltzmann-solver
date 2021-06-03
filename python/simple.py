# %%
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import pandas as pd
import plotly
import scipy as sp
from numpy import ma
from plotly import graph_objects as go

from python.common import *

# %%
# Setup plotting

COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS


# %%
# Find the default output directory
OUTPUT_DIR = Path(gettempdir()) / "boltzmann-solver" / "simple"
print(f"Loading data from {OUTPUT_DIR}")

# %% [markdown]
# # N1F1

# %%
data = read_csv(OUTPUT_DIR / "n1f1.csv")

print("Integration steps:", len(data["n"].index))
print("Final B-L:", data["n"]["ΔB-L"].iloc[-1])
plot_integration(data)


# %%
fig = plot_asymmetry(data, ["H", "L1", "N1"])
fig.update_yaxes(range=[-30, 0])
fig.add_trace(go.Scatter(x=ul["beta"], y=ul["BL"].abs(), name="[U] B-L"))
fig


# %%
fig = plot_density(data, ["H", "L1", "N1"])
fig.add_trace(go.Scatter(x=ul["beta"], y=ul["N1"], name="[U] N1"))
fig.add_trace(go.Scatter(x=ul["beta"], y=ul["N1eq"], name="[U] N1eq"))
# fig.update_yaxes(type="log", range=[-20,1])
fig

# %% [markdown]
# # N1F3

# %%
data = read_csv(OUTPUT_DIR / "n1f3.csv")

print("Integration steps:", len(data["n"].index))
print("Final B-L:", data["n"]["ΔB-L"].iloc[-1])
plot_integration(data)


# %%
plot_asymmetry(data, ["H", "L1", "L2", "L3", "N1"])


# %%
fig = plot_density(data, ["N1"])
fig.update_yaxes(range=[0, 2])

# %% [markdown]
# # N3F3

# %%
data = read_csv(OUTPUT_DIR / "n3f3.csv")

print("Integration steps:", len(data["n"].index))
print("Final B-L:", data["n"]["ΔB-L"].iloc[-1])
plot_integration(data)


# %%
plot_asymmetry(data, ["H", "L1", "L2", "L3", "N1", "N2", "N3"])


# %%
fig = plot_density(data, ["N1", "N2", "N3"])
fig.update_yaxes(range=[0, 2])

# %% [markdown]
# # Miscellaneous
# %% [markdown]
# ## Evolution

# %%
data, ptcls = read_evolution(OUTPUT_DIR / "evolution.json")


# %%
go.Figure(
    data=[go.Scatter(name=p, x=data["beta"], y=data[p, "mass"]) for p in ptcls],
    layout=go.Layout(
        xaxis=go.layout.XAxis(
            title="Inverse Temperature [GeV⁻¹]",
            type="log",
            exponentformat="power",
        ),
        yaxis=go.layout.YAxis(title="Mass [GeV]", type="log", exponentformat="power"),
    ),
)


# %%
go.Figure(
    data=[
        go.Scatter(name=p, x=data["beta"], y=data[p, "mass"] * data["beta"])
        for p in ptcls
    ],
    layout=go.Layout(
        xaxis=go.layout.XAxis(
            title="Inverse Temperature [GeV⁻¹]",
            type="log",
            exponentformat="power",
        ),
        yaxis=go.layout.YAxis(
            title="Mass / Temperatre", type="log", exponentformat="power"
        ),
    ),
)


# %%
go.Figure(
    data=[
        go.Scatter(name=p, x=data["beta"], y=data[p, "width"] / data[p, "mass"])
        for p in ptcls
    ],
    layout=go.Layout(
        xaxis=go.layout.XAxis(
            title="Inverse Temperature [GeV⁻¹]",
            type="log",
            exponentformat="power",
        ),
        yaxis=go.layout.YAxis(
            title="Width / Mass", type="linear", exponentformat="power"
        ),
    ),
)

# %% [markdown]
# ## Gammas

# %%
data = dict()
data["u"] = pd.read_csv("/tmp/josh/ulysses/n1f1.csv")
data["b"] = pd.read_csv(OUTPUT_DIR / "gamma.csv")
go.Figure(
    data=[
        go.Scatter(name="[U] decay", x=data["u"]["beta"], y=data["u"]["d1"]),
        go.Scatter(name="[U] washout", x=data["u"]["beta"], y=data["u"]["w1"]),
        go.Scatter(name="[B] decay", x=data["b"]["beta"], y=data["b"]["N1 ↔ L1 H"]),
        go.Scatter(name="[B] washout", x=data["b"]["beta"], y=data["b"]["L1 ↔ N1"]),
    ],
    layout=go.Layout(
        xaxis=go.layout.XAxis(
            title="Inverse Temperature [GeV⁻¹]",
            type="log",
            exponentformat="power",
        ),
        yaxis=go.layout.YAxis(
            title="Interaction Rate",
            type="log",
            range=[-30, 15],
            exponentformat="power",
        ),
    ),
)


# %%
data = dict()
data["u"] = pd.read_csv("/tmp/josh/ulysses/n1f1.csv")
data["b"] = pd.read_csv(OUTPUT_DIR / "asymmetry.csv")
go.Figure(
    data=[
        go.Scatter(name="[U] decay", x=data["u"]["beta"], y=data["u"]["epsd1"].abs()),
        go.Scatter(
            name="[B] decay", x=data["b"]["beta"], y=data["b"]["N1 ↔ L1 H"].abs()
        ),
    ],
    layout=go.Layout(
        xaxis=go.layout.XAxis(
            title="Inverse Temperature [GeV⁻¹]",
            type="log",
            exponentformat="power",
        ),
        yaxis=go.layout.YAxis(
            title="Interaction Rate", type="log", range=[-40, 5], exponentformat="power"
        ),
    ),
)


# %%
