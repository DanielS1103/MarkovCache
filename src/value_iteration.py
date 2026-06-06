
"""
Algoritmo Value Iteration — R&N Capítulo 16, Figura 16.6.

Resuelve un MDP aplicando iterativamente la actualización de Bellman
hasta alcanzar la convergencia:

    U_{i+1}(s) = max_a  Σ P(s'|s,a) [R(s,a,s') + gamma * U_i(s')]

Criterio de parada (Ecuación 16.12):
    delta < epsilon * (1 - gamma) / gamma

lo que garantiza que el error en la utilidad verdadera es menor que epsilon.
"""

import copy


class ValueIteration:
    """Solver de Value Iteration (R&N Sección 16.2.1, Figura 16.6).

    Args:
        mdp:     instancia de MDP (debe implementar la interfaz MDP)
        epsilon: error máximo permitido en la utilidad de cualquier estado
    """

    def __init__(self, mdp, epsilon=0.001):
        self.mdp = mdp
        self.epsilon = epsilon

    def solve(self):
        """Ejecuta Value Iteration hasta la convergencia.

        Returns:
            U       (dict): estado -> valor de utilidad
            policy  (dict): estado -> mejor acción
            history (list): lista de dicts {estado: utilidad} por iteración,
                            usado para graficar la convergencia
        """
        gamma = self.mdp.get_gamma()
        states = self.mdp.states()

        # Inicializar todas las utilidades en cero
        U = {s: 0.0 for s in states}
        history = []

        # Umbral de convergencia de la Ecuación 16.12
        threshold = self.epsilon * (1.0 - gamma) / gamma

        while True:
            U_prev = copy.copy(U)
            delta = 0.0

            for s in states:
                # Actualización de Bellman: U'[s] = max_a Q(s, a, U_prev)
                nuevo_val = self.mdp.best_q(s, U_prev)
                U[s] = nuevo_val
                diff = abs(U[s] - U_prev[s])
                if diff > delta:
                    delta = diff

            # Guardar snapshot para análisis de convergencia
            history.append(copy.copy(U))

            if delta <= threshold:
                break

        policy = self._extract_policy(U)
        return U, policy, history

    def _extract_policy(self, U):
        """Extrae la política greedy a partir de la función de utilidad U.

        pi(s) = argmax_a Q(s, a, U)
        """
        policy = {}
        for s in self.mdp.states():
            policy[s] = self.mdp.best_action(s, U)
        return policy

    @staticmethod
    def convergence_report(history):
        """Imprime el delta máximo entre snapshots de utilidad sucesivos por iteración."""
        print(f"\n{'Iter':>6}  {'Max |dU|':>12}")
        print("-" * 22)
        for i, U in enumerate(history):
            if i == 0:
                print(f"{i+1:>6}  {'(inicio)':>12}")
                continue
            prev = history[i - 1]
            delta = max(abs(U[s] - prev[s]) for s in U)
            print(f"{i+1:>6}  {delta:>12.8f}")
        print(f"\nConvergió en {len(history)} iteraciones.")