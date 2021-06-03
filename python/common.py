import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import plotly
import plotly.colors
import plotly.graph_objects as go
from IPython.core.display import HTML, display
from ipywidgets import interact, widgets

COLORS = plotly.colors.DEFAULT_PLOTLY_COLORS


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str

    Obtained from https://stackoverflow.com/a/64655638/1573761
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    low_color = (0, 0, 0)
    high_color = (0, 0, 0)
    low_cutoff = 0
    high_cutoff = 1
    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )


def read_number_density(
    f: Union[Path, str], size_threshold: int = 20_000, quiet: bool = False
) -> pd.DataFrame:
    """Read number density CSV file.

    The data is expected to have the following columns:

    - `step`: Integration step
    - `beta`: Inverse temperature [1 / GeV]

    and then all the other columns refer to number densities of the various
    particle species.  For each particle `p`, the three columns should be:

    - `n-p`: Symmetric number density
    - `na-p`: Asymmetric number density
    - `eq-p`: Equilibrium number density

    The output is returned as a dataframe with additional the following columns
    computed:

    - `na-B-L`: The net B-L asymmetric
    - `-na-p`: The negative of the asymmetric number density for all asymmetric
      number densities (including B-L).  This is used when plotting to
      distinguish between positive and negative asymmetries.

    The `size_threshold` option restricts the data size to avoid creating
    unmanageable plots.
    """
    # Read CSV
    data = pd.read_csv(f)
    if len(data.index) > size_threshold:
        data = data.sample(size_threshold)

    data.sort_values("beta", inplace=True)

    # Calculate B-L
    data["na-B-L"] = (1 / 3) * data[
        [
            "na-Q1",
            "na-Q2",
            "na-Q3",
            "na-u1",
            "na-u2",
            "na-u3",
            "na-d1",
            "na-d2",
            "na-d3",
        ]
    ].sum(axis=1)
    data["na-B-L"] -= data[["na-L1", "na-L2", "na-L3", "na-e1", "na-e2", "na-e3"]].sum(
        axis=1
    )
    data["dna-B-L"] = (1 / 3) * data[
        [
            "dna-Q1",
            "dna-Q2",
            "dna-Q3",
            "dna-u1",
            "dna-u2",
            "dna-u3",
            "dna-d1",
            "dna-d2",
            "dna-d3",
        ]
    ].sum(axis=1)
    data["dna-B-L"] -= data[
        ["dna-L1", "dna-L2", "dna-L3", "dna-e1", "dna-e2", "dna-e3"]
    ].sum(axis=1)

    for col in data.columns:
        if col.startswith("na-"):
            data["-" + col] = np.negative(data[col])

    if not quiet:
        display(
            HTML(
                f"""
                <table>
                    <tr>
                        <th>Integration Steps</th>
                        <td>{data["step"].iloc[-1]}</td>
                    </tr>
                    <tr>
                        <th>Plot Samples</th>
                        <td>{len(data.index)}</td>
                    </tr>
                    <tr>
                        <th>Final B-L</th>
                        <td>{data["na-B-L"].iloc[-1]}</td>
                    </tr>
                </table>
                """
            )
        )

    return data


def get_mass_width(ptcls, name) -> List[float]:
    """For the particle's mass and width from a list of particles.

    This is to be used with the JSON-serialized data from a model.
    """
    for ptcl in ptcls:
        if ptcl["name"] == name:
            return [ptcl["mass"], ptcl["width"]]
    raise RuntimeError(f"Particle '{name}' not found")


def read_evolution(filename: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Read JSON-serialized data.

    Returns a tuple with a Pandas DataFrame and a list of particle names.

    The dataframe with the following columns:
    - `beta`: The inverse temperature
    - For each particle:
      - `(p, mass)`: Mass of particle `p`
      - `(p, width)`: Width of particle `p`
    """
    with open(filename, "r") as f:
        data = json.load(f)

    particles = [p["name"] for p in data[0]["sm"]["particles"][1:]]

    df = pd.DataFrame(
        np.array(
            [
                [get_mass_width(d["sm"]["particles"], name) for name in particles]
                for d in data
            ]
        ).reshape((len(data), 2 * len(particles))),
        columns=pd.MultiIndex.from_product([particles, ["mass", "width"]]),
    )
    df["beta"] = [d["sm"]["beta"] for d in data]

    return df, particles


def plot_integration(df: pd.DataFrame):
    """Plot the evolution of the integration.

    Create a plot of inverse temperature against the integration step.
    """
    fig = go.Figure(
        data=[
            go.Scatter(
                name="integration evolution",
                x=df["step"],
                y=df["beta"],
                mode="lines",
            )
        ],
        layout=go.Layout(
            xaxis=go.layout.XAxis(
                title="Integration Step",
                type="linear",
            ),
            yaxis=go.layout.YAxis(
                title="Inverse Temperature [1/GeV]", type="log", exponentformat="power"
            ),
        ),
    )

    return fig


def plot_densities(df: pd.DataFrame, ptcls: List[str]):
    """Plot particle densities.

    The top plot shows the asymmetry of the various particle species listed.  If
    the asymmetry is negative, a dashed line is used instead.

    The bottom shows the symmetric number density and the expected equilibrium
    density.
    """

    fig = plotly.subplots.make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
    )

    fig.add_traces(
        [
            go.Scatter(
                name="B-L",
                legendgroup="B-L",
                showlegend=True,
                x=df["beta"],
                y=df["na-B-L"],
                line=go.scatter.Line(width=3, color="black"),
            ),
            go.Scatter(
                name="-B-L",
                legendgroup="B-L",
                showlegend=True,
                x=df["beta"],
                y=df["-na-B-L"],
                line=go.scatter.Line(width=3, color="black", dash="dash"),
            ),
        ]
        + [
            go.Scatter(
                name=f"{ptcl}",
                legendgroup=ptcl,
                showlegend=False,
                x=df["beta"],
                y=df[f"na-{ptcl}"],
                line=go.scatter.Line(color=color),
            )
            for ptcl, color in zip(ptcls, COLORS)
        ]
        + [
            go.Scatter(
                name=f"-{ptcl}",
                legendgroup=ptcl,
                showlegend=False,
                x=df["beta"],
                y=df[f"-na-{ptcl}"],
                line=go.scatter.Line(color=color, dash="dash"),
            )
            for ptcl, color in zip(ptcls, COLORS)
        ],
        rows=1,
        cols=1,
    )

    fig.add_traces(
        [
            go.Scatter(
                name="B-L",
                legendgroup="B-L",
                showlegend=False,
                x=df["beta"],
                y=df["dna-B-L"].abs(),
                line=go.scatter.Line(width=3, color="black"),
            )
        ]
        + [
            go.Scatter(
                name=f"{ptcl}",
                legendgroup=ptcl,
                showlegend=False,
                x=df["beta"],
                y=df[f"dna-{ptcl}"].abs(),
                line=go.scatter.Line(color=color),
            )
            for ptcl, color in zip(ptcls, COLORS)
        ],
        rows=2,
        cols=1,
    )

    fig.add_traces(
        [
            go.Scatter(
                name=f"{ptcl}",
                legendgroup=ptcl,
                showlegend=False,
                x=df["beta"],
                y=df[f"n-{ptcl}"],
                line=go.scatter.Line(color=color),
            )
            for ptcl, color in zip(ptcls, COLORS)
        ]
        + [
            go.Scatter(
                name=f"eq-{ptcl}",
                legendgroup=ptcl,
                showlegend=False,
                x=df["beta"],
                y=df[f"eq-{ptcl}"],
                line=go.scatter.Line(color=color, dash="dot"),
            )
            for ptcl, color in zip(ptcls, COLORS)
        ],
        rows=3,
        cols=1,
    )

    fig.add_traces(
        [
            go.Scatter(
                name=f"{ptcl}",
                legendgroup=ptcl,
                showlegend=False,
                x=df["beta"],
                y=df[f"dn-{ptcl}"],
                line=go.scatter.Line(color=color),
            )
            for ptcl, color in zip(ptcls, COLORS)
        ],
        rows=4,
        cols=1,
    )

    for row in [1, 2, 3, 4]:
        fig.update_xaxes(
            title_text="Inverse Temperature [1/GeV]" if row == 4 else None,
            type="log",
            exponentformat="power",
            row=row,
            col=1,
        )

    fig.update_yaxes(
        title_text="Asymmetry [Normalized]",
        type="log",
        range=[-20, 1],
        exponentformat="power",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Asymmetry Change [Normalized]",
        type="log",
        range=[-20, 1],
        exponentformat="power",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Density [Normalized]",
        type="linear",
        exponentformat="power",
        rangemode="tozero",
        row=3,
        col=1,
    )
    fig.update_yaxes(
        title_text="Density Change [Normalized]",
        type="linear",
        exponentformat="power",
        rangemode="tozero",
        row=4,
        col=1,
    )

    fig.update_layout(
        width=1000,
        height=1400,
    )

    return fig


def interaction_particle(s: str) -> List[str]:
    return sorted(s.replace("↔", "").replace("\u0304", "").split())


def standardize_interaction(s: str) -> str:
    """Standardize the name of an interaction"""
    return " ".join(interaction_particle(s))


def plot_gamma(data):
    """Plot all the interactions."""
    groups = data.columns[1:].map(standardize_interaction).unique()

    fig = go.FigureWidget(
        layout=go.Layout(
            xaxis=go.layout.XAxis(
                title="Inverse Temperature",
                type="log",
                exponentformat="power",
            ),
            yaxis=go.layout.YAxis(
                title="Interaction Rate",
                type="log",
                exponentformat="power",
                # range=[-20, 20],
            ),
        )
    )
    fig.add_shape(
        type="rect",
        x0=data["beta"].min(),
        x1=data["beta"].max(),
        y0=1e-1,
        y1=1e1,
        fillcolor="Grey",
        line_color="Grey",
        opacity=0.2,
    )

    selected_groups = widgets.SelectMultiple(
        description="Group",
        options=groups,
        value=[],
    )
    particles = widgets.SelectMultiple(
        description="Particle",
        options=sorted(
            set(
                [
                    p.split(".")[0]
                    for group in groups
                    for p in interaction_particle(group)
                ]
            )
        ),
        value=[],
    )

    def update_plot(_):
        with fig.batch_update():
            fig.data = []

            for column in data.columns[1:]:
                if standardize_interaction(column) not in selected_groups.value:
                    continue

                if (
                    not data[column].isnull().all()
                    and data[column].map(lambda x: x > 0).any()
                ):
                    fig.add_trace(
                        go.Scatter(
                            name=column, showlegend=True, x=data["beta"], y=data[column]
                        )
                    )

    def update_groups(_):
        if len(particles.value) == 0:
            selected_groups.options = groups
            return

        new_groups = []
        for group in groups:
            for p in particles.value:
                if p not in group:
                    break
            else:
                new_groups.append(group)
        selected_groups.options = new_groups

    particles.observe(update_groups, names="value")
    selected_groups.observe(update_plot, names="value")

    return widgets.VBox(
        [
            particles,
            selected_groups,
            fig,
        ]
    )
