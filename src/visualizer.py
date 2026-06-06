
"""
Módulo de visualización — todas las gráficas del proyecto usando matplotlib.

Requiere matplotlib (solicitado como librería adicional; únicamente grafica,
no resuelve el MDP ni ningún algoritmo del proyecto).
"""

import os

try:
    import matplotlib
    matplotlib.use("Agg")   # backend no interactivo para guardar PNGs
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("ADVERTENCIA: matplotlib no está instalado. Las gráficas serán omitidas.")


DIRECTORIO_FIGURAS = os.path.join(os.path.dirname(__file__), "..", "figures")


def _asegurar_directorio():
    os.makedirs(DIRECTORIO_FIGURAS, exist_ok=True)


def _guardar(fig, filename):
    _asegurar_directorio()
    path = os.path.join(DIRECTORIO_FIGURAS, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: figures/{filename}")


# ------------------------------------------------------------------
# 1. Curvas de convergencia de Value Iteration
# ------------------------------------------------------------------

def plot_vi_convergence(history, title="Convergencia de Value Iteration",
                        filename="vi_convergencia.png", sample_states=None):
    """Grafica los estimados de utilidad por iteración para estados seleccionados.

    Estilo similar a la Figura 16.7(a) del libro.

    Args:
        history:       lista de dicts {estado: utilidad} por iteración
        title:         título de la gráfica
        filename:      nombre del archivo PNG de salida
        sample_states: lista de estados a graficar (None = seleccionar hasta 6 automáticamente)
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    all_states = list(history[0].keys())

    if sample_states is None:
        # Seleccionar una muestra diversa: hasta 6 estados
        step = max(1, len(all_states) // 6)
        sample_states = all_states[::step][:6]

    fig, ax = plt.subplots(figsize=(9, 5))
    iters = list(range(1, len(history) + 1))

    for state in sample_states:
        values = [h[state] for h in history]
        label = _etiqueta_estado(state)
        ax.plot(iters, values, label=label, linewidth=1.5)

    ax.set_xlabel("Iteración")
    ax.set_ylabel("Estimado de utilidad U(s)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    _guardar(fig, filename)


# ------------------------------------------------------------------
# 2. Delta máximo por iteración
# ------------------------------------------------------------------

def plot_delta_convergence(history, title="Delta máximo por iteración de Bellman",
                           filename="vi_delta.png"):
    """Grafica el cambio máximo en la utilidad por iteración (escala logarítmica)."""
    if not MATPLOTLIB_AVAILABLE:
        return

    deltas = []
    for i in range(1, len(history)):
        delta = max(abs(history[i][s] - history[i - 1][s]) for s in history[i])
        deltas.append(delta)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(2, len(history) + 1), deltas, color="steelblue", linewidth=2)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("max |U_i(s) - U_{i-1}(s)|  (escala log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    _guardar(fig, filename)


# ------------------------------------------------------------------
# 3. Comparación de hit rate entre políticas
# ------------------------------------------------------------------

def plot_hit_rate_comparison(results_by_scenario, filename="comparacion_hit_rate.png"):
    """Gráfica de barras agrupadas: hit rate por escenario y política.

    Args:
        results_by_scenario: dict {nombre_escenario: [SimResult, ...]}
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    scenarios = list(results_by_scenario.keys())
    # Obtener todos los nombres de política (preservando orden del primer escenario)
    policy_names = [r.policy_name for r in next(iter(results_by_scenario.values()))]

    n_policies = len(policy_names)
    x = list(range(len(scenarios)))
    bar_width = 0.8 / n_policies

    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, policy in enumerate(policy_names):
        hit_rates = []
        for scenario in scenarios:
            results = results_by_scenario[scenario]
            match = next((r for r in results if r.policy_name == policy), None)
            hit_rates.append(match.hit_rate if match else 0.0)

        offsets = [xi + (i - n_policies / 2 + 0.5) * bar_width for xi in x]
        ax.bar(offsets, hit_rates, width=bar_width,
               label=policy, color=colors[i % len(colors)], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_ylabel("Hit Rate")
    ax.set_title("Tasa de Aciertos: MDP vs Políticas Baseline")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    _guardar(fig, filename)


# ------------------------------------------------------------------
# 4. Recompensa acumulada a lo largo del tiempo
# ------------------------------------------------------------------

def plot_cumulative_rewards(results, title="Recompensa Acumulada",
                             filename="recompensa_acumulada.png"):
    """Gráfica de líneas de la recompensa acumulada por política a lo largo de los pasos."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors

    for i, r in enumerate(results):
        ax.plot(r.cumulative_rewards, label=r.policy_name,
                color=colors[i % len(colors)], linewidth=1.5)

    ax.set_xlabel("Paso de simulación")
    ax.set_ylabel("Recompensa acumulada")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    _guardar(fig, filename)


# ------------------------------------------------------------------
# 5. Heatmap de utilidades de estados
# ------------------------------------------------------------------

def plot_utility_heatmap(U, mdp, filename="heatmap_utilidades.png"):
    """Heatmap de utilidades agrupadas por configuración de caché.

    Filas = configuraciones de caché (ordenadas por utilidad promedio).
    Columnas = páginas solicitadas.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    states = mdp.states()
    cache_configs = sorted(set(c for c, _ in states))
    pages = sorted(mdp.access_probs.keys())

    # Construir matriz: filas=config_cache, columnas=pagina_solicitada
    matrix = []
    row_labels = []
    for cfg in cache_configs:
        row = [U.get((cfg, p), 0.0) for p in pages]
        matrix.append(row)
        row_labels.append("{" + ",".join(f"P{p}" for p in sorted(cfg)) + "}")

    # Ordenar filas por utilidad promedio (descendente)
    row_means = [sum(row) / len(row) for row in matrix]
    order = sorted(range(len(matrix)), key=lambda i: -row_means[i])
    matrix = [matrix[i] for i in order]
    row_labels = [row_labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(6, len(pages) * 1.2),
                                     max(4, len(cache_configs) * 0.5)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(pages)))
    ax.set_xticklabels([f"P{p}" for p in pages])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Página solicitada")
    ax.set_ylabel("Configuración del caché")
    ax.set_title("Utilidades de los estados U(s)")

    plt.colorbar(im, ax=ax, shrink=0.8)
    _guardar(fig, filename)


# ------------------------------------------------------------------
# 6. Sensibilidad al factor de descuento gamma
# ------------------------------------------------------------------

def plot_gamma_sensitivity(gamma_values, hit_rates_by_gamma,
                            filename="sensibilidad_gamma.png"):
    """Grafica el hit rate de la política MDP en función del factor de descuento gamma."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gamma_values, hit_rates_by_gamma, "o-", color="steelblue",
            linewidth=2, markersize=6)
    ax.set_xlabel("Factor de descuento gamma")
    ax.set_ylabel("Hit rate (política MDP)")
    ax.set_title("Sensibilidad al Factor de Descuento gamma")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    _guardar(fig, filename)


# ------------------------------------------------------------------
# 7. Comparación de convergencia VI vs PI
# ------------------------------------------------------------------

def plot_vi_vs_pi(vi_history, pi_history, filename="vi_vs_pi.png"):
    """Compara la convergencia de Value Iteration y Policy Iteration."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Izquierda: delta de VI por iteración
    vi_deltas = [
        max(abs(vi_history[i][s] - vi_history[i - 1][s]) for s in vi_history[i])
        for i in range(1, len(vi_history))
    ]
    axes[0].semilogy(range(2, len(vi_history) + 1), vi_deltas,
                     color="steelblue", linewidth=2, label="Value Iteration")
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel("max |ΔU| (escala log)")
    axes[0].set_title("Convergencia de Value Iteration")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    # Derecha: cambios de política en PI por iteración
    pi_changes = []
    prev_policy = None
    for policy, U in pi_history:
        if prev_policy is None:
            pi_changes.append(len(policy))
        else:
            pi_changes.append(sum(1 for s in policy if policy[s] != prev_policy[s]))
        prev_policy = policy

    axes[1].bar(range(1, len(pi_changes) + 1), pi_changes,
                color="coral", edgecolor="white", label="Policy Iteration")
    axes[1].set_xlabel("Iteración")
    axes[1].set_ylabel("Estados con cambio de política")
    axes[1].set_title("Convergencia de Policy Iteration")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Value Iteration vs Policy Iteration", fontsize=13)
    plt.tight_layout()
    _guardar(fig, filename)


# ------------------------------------------------------------------
# 8. Entorno no estacionario: UCB vs MDP
# ------------------------------------------------------------------

def plot_nonstationary(ucb_result, mdp_result, switch_step,
                        filename="no_estacionario.png"):
    """Muestra la recompensa acumulada de UCB vs MDP cuando la distribución cambia."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ucb_result.cumulative_rewards, label="UCB (se adapta)",
            color="steelblue", linewidth=1.5)
    ax.plot(mdp_result.cumulative_rewards, label="MDP (política fija)",
            color="coral", linewidth=1.5)
    ax.axvline(x=switch_step, color="gray", linestyle="--", linewidth=1.5,
               label=f"Cambio de distribución (paso {switch_step})")
    ax.set_xlabel("Paso de simulación")
    ax.set_ylabel("Recompensa acumulada")
    ax.set_title("Entorno No Estacionario: UCB vs MDP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _guardar(fig, filename)


# ------------------------------------------------------------------
# Función auxiliar
# ------------------------------------------------------------------

def _etiqueta_estado(state):
    cache, page = state
    return "{" + ",".join(f"P{p}" for p in sorted(cache)) + f"}},req=P{page}"