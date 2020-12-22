# %%
import glob
from pathlib import Path
from tempfile import gettempdir

from python.common import *

# Find the default output directory
OUTPUT_DIR = Path(gettempdir()) / "boltzmann-solver" / "full"
display(HTML(f"<p>Loading data from <code>{OUTPUT_DIR}</p>"))

# %% [markdown]
# # Decay Only
# %% [markdown]
# ## 1 Generation

# %%
data = read_number_density(OUTPUT_DIR / "decay_1.csv")

display(plot_integration(data))
display(plot_densities(data, ["H", "L1", "N1"]))

# %% [markdown]
# ## 3 Generations

# %%
data = read_number_density(OUTPUT_DIR / "decay_3.csv")

display(plot_integration(data))
display(plot_densities(data, ["H", "L1", "L2", "L3", "N1", "N2", "N3"]))

# %% [markdown]
# # Washout Only
# %% [markdown]
# ## 1 Generation

# %%
data = read_csv(OUTPUT_DIR / "washout_1.csv")

print("Integration steps:", len(data["n"].index))
print("Final B-L:", data["n"]["ΔB-L"].iloc[-1])
plot_integration(data)


# %%
plot_asymmetry(data, ["H", "L1", "N1"])


# %%
plot_density(data, ["H", "L1", "N1"])

# %% [markdown]
# ## 3 Generation

# %%
data = read_csv(OUTPUT_DIR / "washout_3.csv")

print("Integration steps:", len(data["n"].index))
print("Final B-L:", data["n"]["ΔB-L"].iloc[-1])
plot_integration(data)


# %%
plot_asymmetry(data, ["H", "L1", "L2", "L3", "N1", "N2", "N3"])


# %%
plot_density(data, ["H", "L1", "L2", "L3", "N1", "N2", "N3"])

# %% [markdown]
# # Full
# %% [markdown]
# ## 1 Generation

# %%
data = read_csv(OUTPUT_DIR / "full_1.csv")

print("Integration steps:", len(data["n"].index))
print("Final B-L:", data["n"]["ΔB-L"].iloc[-1])
plot_integration(data)


# %%
plot_asymmetry(data, ["H", "L1", "N1"])


# %%
plot_density(data, ["H", "L1", "N1"])

# %% [markdown]
# ## 3 Generation

# %%
data = read_csv(OUTPUT_DIR / "full_3.csv")

print("Integration steps:", len(data["n"].index))
print("Final B-L:", data["n"]["ΔB-L"].iloc[-1])
plot_integration(data)


# %%
plot_asymmetry(data, ["H", "L1", "L2", "L3", "N1", "N2", "N3"])


# %%
plot_density(data, ["H", "L1", "L2", "L3", "N1", "N2", "N3"])

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
        yaxis=go.layout.YAxis(title="Width / Mass", type="log", exponentformat="power"),
    ),
)

# %% [markdown]
# ## Higgs Equilibrium

# %%
datas = list(
    map(read_csv, sorted(glob.glob(str(OUTPUT_DIR / "higgs_equilibrium" / "*.csv"))))
)


# %%
go.Figure(
    data=[
        go.Scatter(
            x=data["n"]["beta"],
            y=data["n"]["H"],
            mode="lines",
            line=go.scatter.Line(color=cmap("viridis", i / len(datas))),
            showlegend=False,
        )
        for i, data in enumerate(datas)
    ]
    + [
        go.Scatter(
            x=data["n"]["beta"],
            y=data["n"]["(H)"],
            mode="lines",
            line=go.scatter.Line(color="black"),
            showlegend=False,
        )
        for i, data in enumerate(datas)
    ],
    layout=go.Layout(
        xaxis=go.layout.XAxis(
            title="Inverse Temperature [GeV⁻¹]",
            type="log",
            exponentformat="power",
        ),
        yaxis=go.layout.YAxis(title="Width / Mass", type="log", exponentformat="power"),
    ),
)

# %% [markdown]
# ## Lepton Equilibrium

# %%
datas = list(
    map(read_csv, sorted(glob.glob(str(OUTPUT_DIR / "lepton_equilibrium" / "*.csv"))))
)


# %%
go.Figure(
    data=[
        go.Scatter(
            x=data["n"]["beta"],
            y=data["n"]["L1"],
            mode="lines",
            line=go.scatter.Line(color=cmap("viridis", i / len(datas))),
            showlegend=False,
        )
        for i, data in enumerate(datas)
    ]
    + [
        go.Scatter(
            x=data["n"]["beta"],
            y=data["n"]["(L1)"],
            mode="lines",
            line=go.scatter.Line(color="black"),
            showlegend=False,
        )
        for i, data in enumerate(datas)
    ],
    layout=go.Layout(
        xaxis=go.layout.XAxis(
            title="Inverse Temperature [GeV⁻¹]",
            type="log",
            exponentformat="power",
        ),
        yaxis=go.layout.YAxis(title="Width / Mass", type="log", exponentformat="power"),
    ),
)

# %% [markdown]
# ## Gammas

# %%
data = pd.read_csv(OUTPUT_DIR / "gamma.csv")
data.drop(
    columns=list(filter(lambda x: "3" in x or "2" in x, data.columns)),
    inplace=True,
)
# print(data.columns)
plot_gamma(data)


# %%
data = pd.read_csv(OUTPUT_DIR / "asymmetry.csv")
plot_gamma(data)


# %%
