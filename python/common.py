import numpy as np
import pandas as pd
import json
from io import StringIO
import matplotlib
import matplotlib.cm

import plotly
import plotly.graph_objects as go

COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS


def cmap(name, val):
    colormap = matplotlib.cm.get_cmap(name)
    rgba = colormap(val)
    rgb = tuple(int((255 * x)) for x in rgba[:3])
    return "rgb" + str(rgb)


def read_csv(f):
    data = dict()

    # Read CSV
    data["n"] = pd.read_csv(f)
    if data["n"].index.size > 100_000:
        data["n"] = data["n"].sample(100_000)

    # Calculate B-L
    data["n"]["ΔB-L"] = (1 / 3) * data["n"][
        ["ΔQ1", "ΔQ2", "ΔQ3", "Δu1", "Δu2", "Δu3", "Δd1", "Δd2", "Δd3"]
    ].sum(axis=1)
    data["n"]["ΔB-L"] -= data["n"][["ΔL1", "ΔL2", "ΔL3", "Δe1", "Δe2", "Δe3"]].sum(
        axis=1
    )

    # Create a negative value dataframe
    data["-n"] = data["n"].apply(
        lambda c: c.apply(np.negative) if c.name not in ["step", "beta"] else c
    )

    return data


def get_mass_width(ptcls, name):
    for ptcl in ptcls:
        if ptcl["name"] == name:
            return ptcl["mass"], ptcl["width"]
    raise RuntimeError(f"Particle '{name}' not found")


def read_evolution(f):
    with open(f, "r") as f:
        data = json.load(f)

    particles = [p["name"] for p in data[0]["sm"]["particles"][1:]]

    df = pd.DataFrame(
        np.array(
            [
                [get_mass_width(d["sm"]["particles"], name) for name in particles]
                for d in data
            ]
        ).reshape((1024, 44)),
        columns=pd.MultiIndex.from_product([particles, ["mass", "width"]]),
    )
    df["beta"] = [d["sm"]["beta"] for d in data]

    return df, particles


def plot_integration(data):
    """Plot integration evolution."""
    fig = go.Figure(
        data=[
            go.Scatter(
                name="integration evolution",
                x=data["n"]["step"],
                y=data["n"]["beta"],
                mode="lines",
            )
        ],
        layout=go.Layout(
            xaxis=go.layout.XAxis(title="Integration Step", type="linear",),
            yaxis=go.layout.YAxis(
                title="Inverse Temperature [GeV⁻¹]", type="log", exponentformat="power"
            ),
        ),
    )

    return fig


def plot_asymmetry(data, ptcls):
    """Plot asymmetries.

    Assumes all the data is stored in `data["n"]` and `data["-n"]`.
    """

    fig = go.Figure(
        data=[
            go.Scatter(
                name="ΔB-L",
                x=data["n"]["beta"],
                y=data["n"]["ΔB-L"],
                line=go.scatter.Line(width=3, color="black"),
            ),
            go.Scatter(
                name="ΔB-L",
                showlegend=False,
                x=data["-n"]["beta"],
                y=data["-n"]["ΔB-L"],
                line=go.scatter.Line(width=3, color="black", dash="dash"),
            ),
        ]
        + [
            go.Scatter(
                name=ptcl,
                x=data["n"]["beta"],
                y=data["n"][f"Δ{ptcl}"],
                line=go.scatter.Line(color=color),
            )
            for ptcl, color in zip(ptcls, COLORS)
        ]
        + [
            go.Scatter(
                name=ptcl,
                showlegend=False,
                x=data["-n"]["beta"],
                y=data["-n"][f"Δ{ptcl}"],
                line=go.scatter.Line(dash="dash"),
            )
            for ptcl, color in zip(ptcls, COLORS)
        ],
        layout=go.Layout(
            xaxis=go.layout.XAxis(
                title="Inverse Temperature [GeV⁻¹]", type="log", exponentformat="power"
            ),
            yaxis=go.layout.YAxis(
                title="Asymmetry [Normalized]",
                type="log",
                range=[-20, 1],
                exponentformat="power",
            ),
        ),
    )

    return fig


def plot_density(data, ptcls):
    """Plot number densities

    Assumes all the data is stored in `data["n"]`.
    """

    fig = go.Figure(
        data=[
            go.Scatter(
                name=ptcl,
                x=data["n"]["beta"],
                y=data["n"][ptcl],
                line=go.scatter.Line(color=color),
            )
            for ptcl, color in zip(ptcls, COLORS)
        ]
        + [
            go.Scatter(
                name=ptcl,
                showlegend=False,
                x=data["n"]["beta"],
                y=data["n"][f"({ptcl})"],
                line=go.scatter.Line(color=color, dash="dot"),
            )
            for ptcl, color in zip(ptcls, COLORS)
        ],
        layout=go.Layout(
            xaxis=go.layout.XAxis(
                title="Inverse Temperature [GeV⁻¹]", type="log", exponentformat="power"
            ),
            yaxis=go.layout.YAxis(
                title="Number Density [Normalized]", type="linear", rangemode="tozero",
            ),
        ),
    )

    return fig