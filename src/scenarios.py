
"""
Algoritmo Policy Iteration — R&N Capítulo 16, Figura 16.9.

Alterna entre dos pasos hasta que la política se estabiliza:

  1. Evaluación de política: calcula U = U^pi resolviendo el sistema lineal
         (I - gamma * T_pi) * U = R_pi       (Ecuación 16.14)
     mediante eliminación Gaussiana con pivoteo parcial, implementada desde cero.

  2. Mejora de política: para cada estado, elige la acción que maximiza Q(s,a,U).

Termina cuando ningún estado cambia su acción (la política es óptima).

También incluye Modified Policy Iteration (Sección 16.2.2): en lugar de resolver
el sistema lineal completo, ejecuta k actualizaciones de Bellman simplificadas
bajo la política fija. Es más rápido por iteración pero puede requerir más iteraciones.
"""

import copy


# ======================================================================
# Eliminación Gaussiana (implementada desde cero, sin numpy)
# ======================================================================

def gaussian_elimination(A, b):
    """Resuelve el sistema lineal A*x = b mediante eliminación Gaussiana con pivoteo parcial.

    Args:
        A: lista de listas (matriz n x n)
        b: lista de floats (vector de n elementos)

    Returns:
        x: lista de floats (vector solución)

    Raises:
        ValueError: si el sistema es singular o casi singular
    """
    n = len(b)
    # Construir la matriz aumentada [A | b]
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        # Pivoteo parcial: encontrar la fila con el mayor valor absoluto en esta columna
        fila_max = col
        val_max = abs(M[col][col])
        for fila in range(col + 1, n):
            if abs(M[fila][col]) > val_max:
                val_max = abs(M[fila][col])
                fila_max = fila

        if val_max < 1e-12:
            raise ValueError(f"Matriz singular o casi singular en la columna {col}")

        # Intercambiar la fila pivote
        M[col], M[fila_max] = M[fila_max], M[col]

        pivote = M[col][col]

        # Eliminar por debajo del pivote
        for fila in range(col + 1, n):
            if M[fila][col] == 0.0:
                continue
            factor = M[fila][col] / pivote
            for j in range(col, n + 1):
                M[fila][j] -= factor * M[col][j]

    # Sustitución hacia atrás
    x = [0.0] * n
    for fila in range(n - 1, -1, -1):
        x[fila] = M[fila][n]
        for col in range(fila + 1, n):
            x[fila] -= M[fila][col] * x[col]
        x[fila] /= M[fila][fila]

    return x


# ======================================================================
# Policy Iteration
# ======================================================================

class PolicyIteration:
    """Solver de Policy Iteration (R&N Sección 16.2.2, Figura 16.9).

    Args:
        mdp:         instancia de MDP
        eval_method: 'exact'      - resuelve el sistema lineal (Ecuación 16.14)
                     'iterative'  - Modified Policy Iteration (k actualizaciones de Bellman)
        k:           número de actualizaciones de Bellman para el método 'iterative'
    """

    def __init__(self, mdp, eval_method='exact', k=20):
        self.mdp = mdp
        self.eval_method = eval_method
        self.k = k

    def solve(self):
        """Ejecuta Policy Iteration hasta que la política se estabiliza.

        Returns:
            U       (dict): estado -> valor de utilidad
            policy  (dict): estado -> mejor acción
            history (list): lista de (snapshot de política, snapshot de U) por iteración
        """
        states = self.mdp.states()

        # Inicializar con una política válida (primera acción disponible en cada estado)
        policy = {s: self.mdp.actions(s)[0] for s in states}
        U = {s: 0.0 for s in states}
        history = []

        while True:
            # Paso 1: Evaluación de política
            if self.eval_method == 'exact':
                U = self._policy_evaluation_exact(policy)
            else:
                U = self._policy_evaluation_iterative(policy, U)

            # Paso 2: Mejora de política
            sin_cambios = True
            for s in states:
                mejor_a = self.mdp.best_action(s, U)
                mejor_q = self.mdp.q_value(s, mejor_a, U)
                q_actual = self.mdp.q_value(s, policy[s], U)
                if mejor_q > q_actual + 1e-9:
                    policy[s] = mejor_a
                    sin_cambios = False

            history.append((copy.copy(policy), copy.copy(U)))

            if sin_cambios:
                break

        return U, policy, history

    def _policy_evaluation_exact(self, policy):
        """Resuelve el sistema lineal (I - gamma*T_pi)*U = R_pi.

        Para cada estado s (Ecuación 16.14):
            U(s) = sum_s' P(s'|s,pi(s)) * [R(s,pi(s),s') + gamma*U(s')]

        Reordenando:
            U(s) - gamma * sum_s' P(s'|s,pi(s))*U(s') = sum_s' P(s'|s,pi(s))*R(s,pi(s),s')

        Es decir: (I - gamma*T_pi) * U = r_pi
        """
        states = self.mdp.states()
        n = len(states)
        idx = {s: i for i, s in enumerate(states)}
        gamma = self.mdp.get_gamma()

        # Construir la matriz del lado izquierdo (I - gamma*T_pi) y el vector del lado derecho r_pi
        A = [[0.0] * n for _ in range(n)]
        b = [0.0] * n

        for i, s in enumerate(states):
            a = policy[s]
            A[i][i] = 1.0  # diagonal de la identidad
            for next_s, prob in self.mdp.get_transitions(s, a):
                r = self.mdp.reward(s, a, next_s)
                j = idx[next_s]
                A[i][j] -= gamma * prob   # restar la entrada gamma * T_pi
                b[i] += prob * r          # acumular la recompensa inmediata esperada

        x = gaussian_elimination(A, b)
        return {s: x[i] for i, s in enumerate(states)}

    def _policy_evaluation_iterative(self, policy, U_init):
        """Modified Policy Iteration: k actualizaciones de Bellman bajo política fija.

        U_{t+1}(s) = sum_s' P(s'|s,pi(s)) [R(s,pi(s),s') + gamma*U_t(s')]
        """
        U = copy.copy(U_init)
        states = self.mdp.states()
        gamma = self.mdp.get_gamma()

        for _ in range(self.k):
            U_nuevo = {}
            for s in states:
                a = policy[s]
                val = sum(
                    prob * (self.mdp.reward(s, a, ns) + gamma * U.get(ns, 0.0))
                    for ns, prob in self.mdp.get_transitions(s, a)
                )
                U_nuevo[s] = val
            U = U_nuevo

        return U

    @staticmethod
    def convergence_report(history):
        """Imprime el resumen de cambios de política por iteración."""
        print(f"\n{'Iter':>6}  {'Cambios de política':>20}")
        print("-" * 30)
        prev_policy = None
        for i, (policy, U) in enumerate(history):
            if prev_policy is None:
                cambios = len(policy)
            else:
                cambios = sum(1 for s in policy if policy[s] != prev_policy[s])
            print(f"{i+1:>6}  {cambios:>20}")
            prev_policy = policy
        print(f"\nConvergió en {len(history)} iteraciones.")