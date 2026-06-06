
"""
Clase base abstracta para un Proceso de Decisión de Markov (MDP).
Russell & Norvig, Capítulo 16.

Define la interfaz común que deben implementar todos los MDPs concretos.
Los algoritmos de solución (Value Iteration, Policy Iteration) operan
exclusivamente sobre esta interfaz.
"""


class MDP:
    """Proceso de Decisión de Markov abstracto.

    Las subclases deben implementar: states, actions, get_transitions, reward, get_gamma.
    El método q_value está implementado aquí y es utilizado por todos los algoritmos.
    """

    def states(self):
        """Retorna la lista de todos los estados del MDP."""
        raise NotImplementedError

    def actions(self, state):
        """Retorna la lista de acciones disponibles en el estado dado."""
        raise NotImplementedError

    def get_transitions(self, state, action):
        """Retorna una lista de pares (siguiente_estado, probabilidad) con prob > 0.

        Esta representación dispersa evita iterar sobre todos los estados al
        calcular los Q-valores (la mayoría de las transiciones tienen probabilidad 0).
        """
        raise NotImplementedError

    def reward(self, state, action, next_state):
        """Retorna R(s, a, s') — la recompensa inmediata por la transición."""
        raise NotImplementedError

    def get_gamma(self):
        """Retorna el factor de descuento gamma."""
        raise NotImplementedError

    def q_value(self, state, action, U):
        """Calcula Q(s, a) dados los estimados actuales de utilidad U.

        Función Q-VALUE de R&N página 559:
            Q(s, a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * U[s']]

        Args:
            state:  estado actual s
            action: acción a
            U:      diccionario estado -> estimado de utilidad

        Returns:
            float: Q-valor para el par (estado, acción)
        """
        gamma = self.get_gamma()
        total = 0.0
        for next_state, prob in self.get_transitions(state, action):
            r = self.reward(state, action, next_state)
            total += prob * (r + gamma * U.get(next_state, 0.0))
        return total

    def best_action(self, state, U):
        """Retorna la acción que maximiza Q(s, a) para el U dado."""
        acts = self.actions(state)
        return max(acts, key=lambda a: self.q_value(state, a, U))

    def best_q(self, state, U):
        """Retorna max_a Q(s, a) para el U dado."""
        return max(self.q_value(state, a, U) for a in self.actions(state))