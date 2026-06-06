
"""
Ejecutor principal de experimentos — Proyecto MDP Gestión de Caché.
R&N Capítulo 16: Making Complex Decisions.

Uso:
    python main.py              # ejecutar todos los experimentos
    python main.py --quick      # configuración pequeña para pruebas rápidas
    python main.py --verify     # solo verificaciones de correctitud (sin simulación)
"""

import sys
import time

from src.cache_mdp import CacheMDP
from src.scenarios import (UniformAccess, ZipfAccess,
                            TemporalLocalityAccess, NonStationaryAccess,
                            print_distribution)
from src.value_iteration import ValueIteration
from src.policy_iteration import PolicyIteration
from src.simulator import Simulator
from src.ucb import UCBAgent
import src.visualizer as viz


# ======================================================================
# Configuración
# ======================================================================

QUICK_MODE   = "--quick"  in sys.argv
VERIFY_ONLY  = "--verify" in sys.argv

NUM_PAGES    = 4 if QUICK_MODE else 5   # número total de páginas N
CACHE_SIZE   = 2 if QUICK_MODE else 3   # capacidad del caché K
GAMMA        = 0.9                       # factor de descuento
EPSILON      = 0.001                     # tolerancia de convergencia para VI
SIM_STEPS    = 2000 if QUICK_MODE else 10000  # pasos de simulación
SEED         = 42                        # semilla para reproducibilidad


# ======================================================================
# Funciones auxiliares
# ======================================================================

def seccion(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def build_mdp(dist, gamma=GAMMA):
    """Construye el MDP de caché para la distribución de acceso dada."""
    probs = dist.get_probs()
    return CacheMDP(num_pages=NUM_PAGES, cache_size=CACHE_SIZE,
                    access_probs=probs, gamma=gamma)


def solve_vi(mdp, label=""):
    """Ejecuta Value Iteration y reporta tiempo e iteraciones."""
    t0 = time.time()
    U, policy, history = ValueIteration(mdp, epsilon=EPSILON).solve()
    elapsed = time.time() - t0
    print(f"  VI {label}: {len(history)} iteraciones, {elapsed:.3f}s")
    return U, policy, history


def solve_pi(mdp, label=""):
    """Ejecuta Policy Iteration y reporta tiempo e iteraciones."""
    t0 = time.time()
    U, policy, history = PolicyIteration(mdp, eval_method='exact').solve()
    elapsed = time.time() - t0
    print(f"  PI {label}: {len(history)} iteraciones, {elapsed:.3f}s")
    return U, policy, history


# ======================================================================
# Experimento 0: Verificaciones de correctitud
# ======================================================================

def experiment_verify():
    seccion("EXPERIMENTO 0: Verificaciones de Correctitud")

    dist = ZipfAccess(num_pages=NUM_PAGES, alpha=1.0, seed=SEED)
    mdp = build_mdp(dist)

    print(f"\n  Configuración: N={NUM_PAGES}, K={CACHE_SIZE}, gamma={GAMMA}")
    print(f"  Total de estados: {mdp.num_states()}")

    print("\n  Verificando que las probabilidades de transición suman 1 para todo (s,a)...")
    ok = mdp.verify_transitions()
    print(f"  Resultado: {'CORRECTO' if ok else 'ERROR'}")

    print("\n  Muestra de estados:")
    for state in mdp.states()[:5]:
        print(f"    {mdp.describe_state(state)}")
        for action in mdp.actions(state):
            transitions = mdp.get_transitions(state, action)
            total_p = sum(p for _, p in transitions)
            print(f"      acción={mdp.describe_action(action)}, "
                  f"transiciones={len(transitions)}, suma_p={total_p:.6f}")

    print("\n  Verificando que VI y PI producen la misma política óptima...")
    U_vi, pol_vi, _ = solve_vi(mdp, "verificar")
    U_pi, pol_pi, _ = solve_pi(mdp, "verificar")

    discrepancias = [s for s in mdp.states() if pol_vi[s] != pol_pi[s]]
    print(f"  Discrepancias en política: {len(discrepancias)} / {mdp.num_states()}")
    if discrepancias:
        print("  ADVERTENCIA: las políticas difieren. Revisar la implementación.")
    else:
        print("  VI y PI coinciden en la política óptima. CORRECTO")


# ======================================================================
# Experimento 1: Convergencia de Value Iteration por escenario
# ======================================================================

def experiment_vi_convergence():
    seccion("EXPERIMENTO 1: Convergencia de Value Iteration")

    distributions = [
        UniformAccess(num_pages=NUM_PAGES, seed=SEED),
        ZipfAccess(num_pages=NUM_PAGES, alpha=1.0, seed=SEED),
        TemporalLocalityAccess(num_pages=NUM_PAGES, num_hot_pages=CACHE_SIZE,
                               hot_prob=0.8, seed=SEED),
    ]

    for dist in distributions:
        print(f"\n  Distribución: {dist.name()}")
        print_distribution(dist)
        mdp = build_mdp(dist)
        U, policy, history = solve_vi(mdp, dist.name())

        # Imprimir política óptima para estados de miss de muestra
        miss_states = [(c, p) for c, p in mdp.states() if p not in c][:6]
        print("\n  Política óptima (estados de miss):")
        for state in miss_states:
            print(f"    {mdp.describe_state(state):<45} -> "
                  f"{mdp.describe_action(policy[state])}")

        # Generar gráficas
        viz.plot_vi_convergence(
            history,
            title=f"Convergencia VI — {dist.name()}",
            filename=f"vi_convergencia_{dist.__class__.__name__}.png",
        )
        viz.plot_delta_convergence(
            history,
            title=f"Delta por Iteración — {dist.name()}",
            filename=f"vi_delta_{dist.__class__.__name__}.png",
        )
        viz.plot_utility_heatmap(U, mdp,
                                  filename=f"heatmap_utilidades_{dist.__class__.__name__}.png")


# ======================================================================
# Experimento 2: Value Iteration vs Policy Iteration
# ======================================================================

def experiment_vi_vs_pi():
    seccion("EXPERIMENTO 2: Value Iteration vs Policy Iteration")

    dist = ZipfAccess(num_pages=NUM_PAGES, alpha=1.0, seed=SEED)
    mdp = build_mdp(dist)

    print(f"\n  Distribución: {dist.name()}")
    U_vi, pol_vi, vi_history = solve_vi(mdp, "Zipf")
    U_pi, pol_pi, pi_history = solve_pi(mdp, "Zipf")

    ValueIteration.convergence_report(vi_history)
    PolicyIteration.convergence_report(pi_history)

    viz.plot_vi_vs_pi(vi_history, pi_history, filename="vi_vs_pi.png")


# ======================================================================
# Experimento 3: Comparación de políticas en simulación
# ======================================================================

def experiment_policy_comparison():
    seccion("EXPERIMENTO 3: Comparación de Políticas (MDP vs Baselines)")

    distributions = [
        ("Uniforme",  UniformAccess(num_pages=NUM_PAGES, seed=SEED)),
        ("Zipf",      ZipfAccess(num_pages=NUM_PAGES, alpha=1.0, seed=SEED)),
        ("Temporal",  TemporalLocalityAccess(num_pages=NUM_PAGES,
                                             num_hot_pages=CACHE_SIZE,
                                             hot_prob=0.8, seed=SEED)),
    ]

    results_by_scenario = {}

    for name, dist in distributions:
        print(f"\n  Escenario: {name}")
        mdp = build_mdp(dist)

        _, pol_vi, _ = solve_vi(mdp, name)
        _, pol_pi, _ = solve_pi(mdp, name)

        ucb = UCBAgent(cache_size=CACHE_SIZE, num_pages=NUM_PAGES,
                       c=1.414, seed=SEED)
        sim = Simulator(NUM_PAGES, CACHE_SIZE, dist, seed=SEED)

        results = sim.run_all(
            mdp_vi_policy=pol_vi,
            mdp_pi_policy=pol_pi,
            ucb_agent=ucb,
            num_steps=SIM_STEPS,
            include_optimal=True,
        )
        results_by_scenario[name] = results
        Simulator.print_results(results)

    viz.plot_hit_rate_comparison(results_by_scenario,
                                  filename="comparacion_hit_rate.png")

    # Recompensa acumulada para el escenario Zipf (el más ilustrativo)
    zipf_results = results_by_scenario["Zipf"]
    viz.plot_cumulative_rewards(zipf_results,
                                 title="Recompensa Acumulada — Distribución Zipf",
                                 filename="recompensa_acumulada_zipf.png")


# ======================================================================
# Experimento 4: Sensibilidad al factor de descuento gamma
# ======================================================================

def experiment_gamma_sensitivity():
    seccion("EXPERIMENTO 4: Análisis de Sensibilidad a Gamma")

    dist = ZipfAccess(num_pages=NUM_PAGES, alpha=1.0, seed=SEED)
    gammas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    hit_rates = []

    for gamma in gammas:
        mdp = build_mdp(dist, gamma=gamma)
        _, policy, _ = solve_vi(mdp, f"gamma={gamma}")
        sim = Simulator(NUM_PAGES, CACHE_SIZE, dist, seed=SEED)
        result = sim.simulate_mdp(policy, num_steps=SIM_STEPS, policy_name="MDP-VI")
        hit_rates.append(result.hit_rate)
        print(f"  gamma={gamma:.2f}  hit_rate={result.hit_rate:.4f}")

    viz.plot_gamma_sensitivity(gammas, hit_rates, filename="sensibilidad_gamma.png")


# ======================================================================
# Experimento 5: Escalabilidad del espacio de estados
# ======================================================================

def experiment_scalability():
    seccion("EXPERIMENTO 5: Escalabilidad (Tamaño del Espacio de Estados vs Tiempo)")

    configs = [(4, 2), (5, 3), (6, 3), (7, 3), (8, 3)]
    print(f"\n  {'N':>4}  {'K':>4}  {'|S|':>8}  {'Iter VI':>10}  {'Tiempo VI(s)':>13}  {'Iter PI':>10}  {'Tiempo PI(s)':>13}")
    print("  " + "-" * 72)

    for n, k in configs:
        dist = ZipfAccess(num_pages=n, alpha=1.0, seed=SEED)
        probs = dist.get_probs()
        mdp = CacheMDP(num_pages=n, cache_size=k, access_probs=probs, gamma=GAMMA)
        n_states = mdp.num_states()

        t0 = time.time()
        _, _, vi_hist = ValueIteration(mdp, epsilon=EPSILON).solve()
        vi_time = time.time() - t0

        t0 = time.time()
        _, _, pi_hist = PolicyIteration(mdp, eval_method='exact').solve()
        pi_time = time.time() - t0

        print(f"  {n:>4}  {k:>4}  {n_states:>8}  "
              f"{len(vi_hist):>10}  {vi_time:>13.3f}  "
              f"{len(pi_hist):>10}  {pi_time:>13.3f}")


# ======================================================================
# Experimento 6: Entorno no estacionario (UCB vs MDP)
# ======================================================================

def experiment_nonstationary():
    seccion("EXPERIMENTO 6: Entorno No Estacionario")

    SWITCH = SIM_STEPS // 2

    dist1 = ZipfAccess(num_pages=NUM_PAGES, alpha=1.0, seed=SEED)

    # Fase 2: invertir la popularidad de las páginas (la menos popular se vuelve la más popular)
    probs1 = dist1.get_probs()
    pages_by_pop = sorted(probs1.keys(), key=lambda p: probs1[p], reverse=True)
    reversed_probs = {p: probs1[pages_by_pop[-(i + 1)]] for i, p in enumerate(pages_by_pop)}

    dist2 = ZipfAccess(num_pages=NUM_PAGES, alpha=1.0, seed=SEED)
    dist2._probs = reversed_probs  # sobreescribir con popularidad invertida

    nonstat_dist = NonStationaryAccess(dist1, dist2, switch_step=SWITCH, seed=SEED)

    print(f"\n  Fase 1 (pasos 0-{SWITCH-1}): {dist1.name()}")
    print(f"  Fase 2 (pasos {SWITCH}-{SIM_STEPS-1}): Zipf con popularidad invertida")

    # El MDP es entrenado con la distribución de la fase 1 (no conoce el cambio)
    mdp = build_mdp(dist1)
    _, mdp_policy, _ = solve_vi(mdp, "dist1")

    ucb = UCBAgent(cache_size=CACHE_SIZE, num_pages=NUM_PAGES, c=1.414, seed=SEED)

    sim = Simulator(NUM_PAGES, CACHE_SIZE, nonstat_dist, seed=SEED)
    mdp_result = sim.simulate_mdp(mdp_policy, num_steps=SIM_STEPS,
                                   policy_name="MDP (fijo)")
    ucb_result = sim.simulate_ucb(ucb, num_steps=SIM_STEPS)

    print(f"\n  MDP (fijo):   hit_rate={mdp_result.hit_rate:.4f}")
    print(f"  UCB (online): hit_rate={ucb_result.hit_rate:.4f}")

    viz.plot_nonstationary(ucb_result, mdp_result, SWITCH,
                            filename="no_estacionario.png")


# ======================================================================
# Punto de entrada
# ======================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Simulador de Gestión de Caché con MDPs")
    print("# R&N Capítulo 16 — Making Complex Decisions")
    print(f"# Config: N={NUM_PAGES}, K={CACHE_SIZE}, gamma={GAMMA}, pasos={SIM_STEPS}")
    print("#" * 60)

    experiment_verify()

    if VERIFY_ONLY:
        print("\nModo verificación: finalizado.")
        sys.exit(0)

    experiment_vi_convergence()
    experiment_vi_vs_pi()
    experiment_policy_comparison()
    experiment_gamma_sensitivity()
    experiment_scalability()
    experiment_nonstationary()

    print("\n" + "=" * 60)
    print("  Todos los experimentos completados.")
    print("  Figuras guardadas en: figures/")
    print("=" * 60)