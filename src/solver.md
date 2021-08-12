Solver for the number density evolution given by integrating the Boltzmann
equation.

Solving the Boltzmann equation in generality without making any assumption about
the phase space distribution `$f$` (other than specifying some initial
condition) is a difficult problem. Furthermore, within the context of
baryogenesis and leptogenesis, we are only interested in the number density (or
more specifically, the difference in the number densities or a particle and its
corresponding antiparticle).

# Assumptions

A couple of assumptions can be made to simplify the Boltzmann equation so that
the number densities can be computed directly.

- _Assume kinetic equilibrium._ If the rate at which a particle species is
  scattering is sufficiently fast, phase space distribution of this species will
  rapidly converge converge onto either the Bose–Einstein or Fermi–Dirac
  distributions:

  ```math
  f_{\text{BE}} = \frac{1}{\exp[(E - \mu) \beta] - 1}, \qquad
  f_{\text{FD}} = \frac{1}{\exp[(E - \mu) \beta] + 1}.
  ```

  For a particular that remains in kinetic equilibrium, the evolution of its
  phase space is entirely described by the evolution of `$\mu$` in time.

- _Assume `$\beta \gg E - \mu$`._ In the limit that the temperature is much less
  than `$E - \mu$` (or equivalently, that the inverse temperature `$\beta$` is
  greater than `$E - \mu$`), both the Fermi–Dirac and Bose–Einstein approach the
  Maxwell–Boltzmann distribution,

  ```math
  f_{\text{MB}} = \exp[-(E - \mu) \beta].
  ```

  This simplifies the expression for the number density to

  ```math
  n = \frac{g m^2 K_2(m \beta)}{2 \pi^2 \beta} e^{\mu \beta} = n^{(0)} e^{\mu \beta},
  ```

  where `$n^{(0)}$` is the equilibrium number density when `$\mu = 0$`. This
  also allows for the equilibrium phase space distribution to be expressed in
  terms of the `$\mu = 0$` distribution:

  ```math
  f_{\text{MB}} = e^{\mu \beta} f_{\text{MB}}^{(0)} = \frac{n}{n^{(0)}} f_{\text{MB}}^{(0)}.
  ```

  Furthermore, the assumption that `$\beta \gg E - \mu$` implies that
  `$f_{\text{BE}}, f_{\text{FD}} \ll 1$`. Consequently, the Pauli suppression
  and Bose enhancement factors in the collision term can all be neglected
  resulting in:

  ```math
  \begin{aligned}
    g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
      &= - \int_{\vt a}^{\vt b}
         \abs{\scM(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} f_i \right)
         - \abs{\scM(\vt b \to \vt a)}^2 \left( \prod_{i \in \vt b} f_i \right) \\\\
      &= - \int_{\vt a}^{\vt b}
         \abs{\scM(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} \frac{n_i}{n_i^{(0)}} f_i^{(0)} \right)
          - \abs{\scM(\vt b \to \vt a)}^2 \left( \prod_{i \in \vt b} \frac{n_i}{n_i^{(0)}} f_i^{(0)} \right).
  \end{aligned}
  ```

  This is then commonly expressed as,

  ```math
  g_{a_1} \int \vt C[f_{a_{1}}] \frac{\dd \vt p_{a_1}}{(2\pi)^3}
    = - \left( \prod_{i \in \vt a} \frac{n_i}{n_i^{(0)}} \right) \gamma(\vt a \to \vt b)
      + \left( \prod_{i \in \vt b} \frac{n_i}{n_i^{(0)}} \right) \gamma(\vt b \to \vt a),
  ```

  where we have introduced the interaction density

  ```math
  \gamma(\vt a \to \vt b)
    = \int_{\vt a}^{\vt b}
      \abs{\scM(\vt a \to \vt b)}^2 \left( \prod_{i \in \vt a} f^{(0)}_i \right).
  ```

# Interactions

The majority of interactions can be accounted for by looking at simply `$1 \leftrightarrow 2$` decays/inverse decays and `$2 \leftrightarrow 2$`
scatterings. In the former case, the phase space integration can be done
entirely analytically and is presented below; whilst in the latter case, the
phase space integration can be simplified to just two integrals.

## Decays and Inverse Decays

Considering the interaction `$a \leftrightarrow b + c$`, the reaction is

```math
\begin{aligned}
  g_a \int \vt C[f_a] \frac{\dd \vt p_a}{(2\pi)^3}
    &= - \frac{n_a}{n^{(0)}_a} \gamma(a \to bc) + \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} \gamma(bc \to a)
\end{aligned}
```

In every case, the squared matrix element will be completely independent of
all initial and final state momenta (even `$\varepsilon \cdot p$` terms
vanish after averaging over both initial and final spins).

### Decay Term

For the decay, the integration over the final state momenta is trivial as
the only final-state-momentum dependence appears within the Dirac delta
function. Consequently, we need only integrate over the initial state
momentum:

```math
\begin{aligned}
  \gamma(a \to bc)
    &= - \abs{\scM(a \to bc)}^2 \int_{a}^{b,c} f^{(0)}_a \\\\
    &= - \frac{\abs{\scM(a \to bc)}^2}{16 \pi} \frac{m_a K_1(m_a \beta)}{2 \pi^2 \beta}
\end{aligned}
```

Combined with the number density scaling:

```math
\frac{n_a}{n^{(0)}_a} \gamma(a \to bc)
  = n_a \frac{\abs{\scM(a \to bc)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
```

where the analytic expression for `$n^{(0)}_a$` was used to introduce the
second Bessel function.

The ratio of Bessel functions is generally referred to as the _time-dilation
factor_ for the decaying particle. When `$m \beta \gg 1$`, the ratio of
Bessel functions approaches 1, and when `$m \beta \ll 1$`, the ratio
approaches 0.

If the final state particles are essentially massless in comparison to the
decaying particle, then the decay rate in the rest frame of the particle of
`$\Gamma_\text{rest} = \abs{\scM}^2 / 16 \pi m_a$` and the above expression
can be simplified to

```math
\frac{n_a}{n^{(0)}_a} \gamma(a \to bc)
= n_a \Gamma_\text{rest} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}.
```

### Inverse Decay

The inverse decay rate is given by

```math
\gamma(bc \to a)
  = \abs{\scM(bc \to a)}^2 \int_{a}^{b,c} f^{(0)}_b f^{(0)}_c
```

The Dirac delta enforces that `$E_a = E_b + E_c$` which implies that
`$f^{(0)}_a = f^{(0)}_b f^{(0)}_c$` thereby making the integral identical to
the decay scenario:

```math
\gamma(bc \to a)
  = \frac{\abs{\scM(bc \to a)}^2}{16 \pi} \frac{m_a K_1(m_a \beta)}{2 \pi^2 \beta}
  = n^{(0)}_a \frac{\abs{\scM(bc \to a)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
```

The full expression including the number density scaling becomes:

```math
\frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} \gamma(bc \to a)
  = \frac{n_b n_c}{n^{(0)}_b n^{(0)}_c} n^{(0)}_a \frac{\abs{\scM(bc \to a)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)}
```

### Combined Decay and Inverse Decay

If only the tree-level contributions are considered for the decay and
inverse decays, then both squared amplitudes will be in fact identical and
thus `$\gamma(ab \to c) \equiv \gamma(c \to ab)$`. As a result, the decay
and inverse decay only differ by a scaling factor to take into account the
initial state particles.

In particular, we can define an alternative reaction rate,

```math
\tilde \gamma(a \to bc) = \frac{\abs{\scM(a \to bc)}^2}{16 \pi m_a} \frac{K_1(m_a \beta)}{K_2(m_a \beta)},
```

which allows for the overall `$1 \leftrightarrow 2$` reaction rate to be
expressed as:

```math
\frac{n_a}{n_a^{(0)}} \gamma(a \to bc) - \frac{n_b n_c}{n_b^{(0)} n_c^{(0)}} \gamma(bc \to a)
= \left[ n_a - \frac{n_b n_c}{n_b^{(0)} n_c^{(0)}} n_a^{(0)} \right] \tilde \gamma(a \to bc)
```

provided the forward and backward rates are equal.

## Two Body Scattering

The two-to-two scattering `$ab \to cd$` reaction density is given by

```math
\gamma(ab \to cd) = \int_{a,b}^{c,d} \abs{\scM(ab \to cd)}^2 f_a^{(0)} f_b^{(0)}
```

The two initial-state phase space integrals can be reduced to a simple
integral over the centre-of-mass `$s$`:

```math
\gamma(ab \to cd) = \frac{1}{8 \pi^3 \beta} \int \hat \sigma_{ab}^{cd}(s) \sqrt{s} K_1(\sqrt{s} \beta) \dd s
```

where `$\hat \sigma(s)$` is the reduced cross-section:

```math
\hat \sigma_{ab}^{cd}(s) = \frac{g_a g_b g_c g_d}{64 \pi^2 s} \int \abs{\scM(ab \to cd)}^2 \dd t
```

in which `$t$` is the usual Mandelstam variable.

As a result, the full `$2 \to 2$` cross-section can be expressed as

```math
\gamma(ab \to cd) = \frac{g_a g_b g_c g_d}{512 \pi^5 \beta} \int \abs{\scM(ab \to cd)}^2 \frac{K_1(\sqrt s \beta)}{\sqrt s} \dd s \dd t
```

### Real Intermediate State

When considering both `$1 \leftrightarrow 2$` and `$2 \leftrightarrow 2$`
scattering processes, there is a double counting issue that arises from
having a real intermediate state (RIS) in `$2 \leftrightarrow 2$`
interactions.

As a concrete example, one may have the processes `$ab \leftrightarrow X$`
and `$X \leftrightarrow cd$` and thus also `$ab \leftrightarrow cd$`. In
computing the squared amplitude for the `$2 \leftrightarrow 2$` process, one
needs to subtract the RIS:

```math
\abs{\scM(ab \leftrightarrow cd)}^2 = \abs{\scM_\text{full}(ab \leftrightarrow cd)}^2
- \abs{\scM_\text{RIS}(ab \leftrightarrow cd)}^2
```

In the case of a single scalar RIS, the RIS-subtracted amplitude is given by

```math
\begin{aligned}
  \abs{\scM_\text{RIS}(ab \to cd)}
  &= \frac{\pi}{m_X \Gamma_X} \delta(s - m_X^2) \theta(\sqrt{s}) \\
  &\quad \Big[
      \abs{\scM(ab \to X)}^2 \abs{\scM(X \to cd)}^2
    + \abs{\scM(ab \to \overline X)}^2 \abs{\scM(\overline X \to cd)}^2
  \Big].
\end{aligned}
```

Care must be taken for fermions as the spinorial structure prevents a simple
factorization in separate squared amplitude. Furthermore if there are
multiple intermediate states, the mixing between these states must also be
taken into account.

# Temperature Evolution

Although it is most intuitive to describe the interaction rates in terms of
time, most quantities depend on the temperature of the Universe at a
particular time. We therefore make the change of variables from `$t$` to
`$\beta$` which introduces a factor of `$H(\beta) \beta$`:

```math
\pfrac{n}{t} + 3 H n \equiv H \beta \pfrac{n}{\beta} + 3 H n.
```

As a result, the actual change in the number density, it becomes

```math
\pfrac{n}{\beta} = \frac{1}{H \beta} \left[\vt C[n] - 3 H n\right]
```

and one must only input `$\vt C[n]$` in the interaction.

# Normalized Number Density

The number densities themselves can be often quite large (especially in the
early Universe), and often are compared to other number densities; as a
result, they are often normalized to either the photon number density
`$n_\gamma$` or the entropy density `$s$`. This library uses a equilibrium
number density of a single massless bosonic degree of freedom (and thus
differs from using the photon number density by a factor of 2).

When dealing with number densities, the Liouville operator is:

```math
\pfrac{n}{t} + 3 H n = \vt C[n]
```

where `$\vt C[n]$` is the change in the number density. If we now define

```math
Y \defeq \frac{n}{n_{\text{eq}}},
```

then the change in this normalized number density is:

```math
\pfrac{Y}{t} = \frac{1}{n_{\text{eq}}} \vt C[n]
```

with `$n_{\text{eq}}$` having the simple analytic form `$3 \zeta(3) / 4 \pi^2 \beta^3$`. Furthermore, make the change of variable from time `$t$`
to inverse temperature `$\beta$`, we get:

```math
\pfrac{Y}{\beta} = \frac{1}{H \beta n_{\text{eq}}} \vt C[n].
```

As with the non-normalized number density calculations, only `$\vt C[n]$`
must be inputted in the interaction.
