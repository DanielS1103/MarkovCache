
"""
Gestión de caché modelada como un MDP — R&N Capítulo 16.

ESTADO: estado aumentado (cache_contents: frozenset, requested_page: int)
  - cache_contents: páginas actualmente en caché (tamaño K)
  - requested_page: página que la CPU acaba de solicitar

ACCIÓN:
  - "noop"          si la página solicitada ya está en caché (hit)
  - ("evict", p)    para cada página p en caché, cuando hay un miss

TRANSICIÓN P(s'|s,a):
  Tras ejecutar la acción a, el caché resultante C' queda determinado.
  El siguiente estado es (C', q) con probabilidad access_probs[q] para cada página q.

RECOMPENSA:
  +1 si la página solicitada está en caché (hit)
  -1 si la página solicitada no está en caché (miss)
"""

import itertools
from src.mdp import MDP


class CacheMDP(MDP):
    """Gestión de caché modelada como un MDP completamente observable.

    Args:
        num_pages:    número total de páginas distintas (N)
        cache_size:   cantidad de páginas que caben en caché (K), K < N
        access_probs: diccionario {page_id: probabilidad}, debe sumar 1.0
        gamma:        factor de descuento en [0, 1)
    """

    def __init__(self, num_pages, cache_size, access_probs, gamma=0.9):
        assert cache_size < num_pages, "El caché debe ser más pequeño que el total de páginas"
        assert abs(sum(access_probs.values()) - 1.0) < 1e-9, "Las probabilidades deben sumar 1"

        self.num_pages = num_pages
        self.cache_size = cache_size
        self.access_probs = access_probs
        self._gamma = gamma
        self.page_ids = sorted(access_probs.keys())

        self._states = None        # se construye de forma lazy
        self._state_index = None   # estado -> índice entero (para Policy Iteration)

    # ------------------------------------------------------------------
    # Interfaz MDP
    # ------------------------------------------------------------------

    def states(self):
        if self._states is None:
            self._build_states()
        return self._states

    def actions(self, state):
        """Retorna las acciones disponibles para el estado aumentado dado."""
        cache, requested = state
        if requested in cache:
            return ["noop"]
        return [("evict", p) for p in sorted(cache)]

    def get_transitions(self, state, action):
        """Retorna la lista dispersa de (siguiente_estado, prob) con prob > 0.

        Desde cualquier (caché, página_solicitada) tras la acción a:
          1. Se calcula el caché resultante C' (determinístico dado a)
          2. El siguiente estado es (C', q) para cada página q, con prob access_probs[q]
        Solo existen N transiciones no nulas (una por cada posible solicitud futura).
        """
        cache, requested = state
        if self._states is None:
            self._build_states()

        # Determinar el caché resultante después de la acción
        if action == "noop":
            result_cache = cache
        else:
            _, evicted = action
            result_cache = (cache - {evicted}) | {requested}

        # El siguiente estado depende de qué página se solicite a continuación
        transitions = []
        for q in self.page_ids:
            prob = self.access_probs[q]
            if prob > 0:
                next_state = (result_cache, q)
                transitions.append((next_state, prob))
        return transitions

    def reward(self, state, action, next_state):
        """R(s, a, s') = +1 en hit, -1 en miss. Depende solo del estado actual."""
        cache, requested = state
        return 1.0 if requested in cache else -1.0

    def get_gamma(self):
        return self._gamma

    # ------------------------------------------------------------------
    # Construcción del espacio de estados
    # ------------------------------------------------------------------

    def _build_states(self):
        """Genera todos los estados aumentados (frozenset de tamaño K, page_id).

        Total de estados = C(N, K) * N.
        Con N=5, K=3: C(5,3)*5 = 50 estados.
        Con N=8, K=3: C(8,3)*8 = 448 estados.
        """
        all_cache_configs = [
            frozenset(combo)
            for combo in itertools.combinations(self.page_ids, self.cache_size)
        ]
        self._states = [
            (cache, page)
            for cache in all_cache_configs
            for page in self.page_ids
        ]
        self._state_index = {s: i for i, s in enumerate(self._states)}

    # ------------------------------------------------------------------
    # Utilidades auxiliares
    # ------------------------------------------------------------------

    def state_index(self, state):
        """Retorna el índice entero de un estado (usado por el solver lineal de Policy Iteration)."""
        if self._state_index is None:
            self._build_states()
        return self._state_index[state]

    def num_states(self):
        return len(self.states())

    def hit_probability(self, state):
        """Probabilidad de que una solicitud aleatoria resulte en hit para el caché actual."""
        cache, _ = state
        return sum(self.access_probs[p] for p in cache)

    def describe_state(self, state):
        """Representación legible de un estado."""
        cache, requested = state
        hit = "HIT" if requested in cache else "MISS"
        cache_str = "{" + ", ".join(f"P{p}" for p in sorted(cache)) + "}"
        return f"cache={cache_str}, req=P{requested} [{hit}]"

    def describe_action(self, action):
        """Representación legible de una acción."""
        if action == "noop":
            return "noop (mantener caché)"
        _, evicted = action
        return f"desalojar P{evicted}"

    def describe_policy(self, policy):
        """Imprime la política completa como tabla legible."""
        print(f"\n{'ESTADO':<45} {'ACCIÓN'}")
        print("-" * 65)
        for state in self.states():
            act = policy.get(state, "?")
            print(f"{self.describe_state(state):<45} {self.describe_action(act)}")

    def verify_transitions(self):
        """Verificación de correctitud: las probabilidades de transición suman 1 para todo (s,a)."""
        errors = []
        for state in self.states():
            for action in self.actions(state):
                total = sum(p for _, p in self.get_transitions(state, action))
                if abs(total - 1.0) > 1e-9:
                    errors.append((state, action, total))
        if errors:
            for s, a, t in errors:
                print(f"ERROR: suma={t:.6f} para estado={s}, acción={a}")
            return False
        return True