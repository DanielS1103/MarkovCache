
"""
Agente UCB1 (Upper Confidence Bound) para gestión de caché online — R&N Sección 16.3.

Modela cada decisión de desalojo como el brazo de un bandido multi-brazo.
No requiere conocer la distribución de acceso a páginas con antelación;
aprende de la experiencia a través de la exploración y explotación.

Índice UCB para el par (estado, acción):
    UCB(s, a) = Q(s, a) + c * sqrt(ln(N_s) / N_sa)

donde:
    Q(s, a)  = recompensa promedio observada cuando se eligió (s, a)
    N_s      = número total de veces que se visitó el estado s
    N_sa     = número de veces que se tomó la acción a en el estado s
    c        = parámetro de exploración (mayor valor = más exploración)

En estados de hit la única acción es 'noop' (sin decisión que tomar),
por lo que UCB solo aplica cuando hay un miss con caché llena.
"""

import math
import random


class UCBAgent:
    """Agente online UCB1 para decisiones de desalojo en caché.

    Args:
        cache_size: K (número de páginas que caben en caché)
        num_pages:  N (total de páginas distintas)
        c:          coeficiente de exploración (sqrt(2) es el estándar de UCB1)
        seed:       semilla aleatoria para desempate
    """

    def __init__(self, cache_size, num_pages, c=1.414, seed=None):
        self.cache_size = cache_size
        self.num_pages = num_pages
        self.c = c
        self._rng = random.Random(seed)

        # Q[clave_estado][clave_accion] = recompensa promedio
        self._Q = {}
        # N_sa[clave_estado][clave_accion] = contador de visitas
        self._N_sa = {}
        # N_s[clave_estado] = total de visitas al estado
        self._N_s = {}

    # ------------------------------------------------------------------
    # Helpers para claves de estado y acción
    # ------------------------------------------------------------------

    def _skey(self, cache, requested_page):
        """Clave hasheable para un estado (frozenset del caché, página solicitada)."""
        return (frozenset(cache), requested_page)

    def _akey(self, action):
        return action  # ya es hasheable ("noop" o ("evict", p))

    # ------------------------------------------------------------------
    # Interfaz principal del agente
    # ------------------------------------------------------------------

    def select_action(self, cache, requested_page):
        """Elige una acción de desalojo usando UCB1.

        Si la página solicitada ya está en caché, retorna "noop" directamente.
        En caso contrario, aplica UCB1 para elegir qué página desalojar.
        """
        if requested_page in cache:
            return "noop"

        acciones_disponibles = [("evict", p) for p in sorted(cache)]
        skey = self._skey(cache, requested_page)
        N_s = self._N_s.get(skey, 0)

        # Siempre explorar primero las acciones no visitadas
        no_visitadas = [
            a for a in acciones_disponibles
            if self._N_sa.get(skey, {}).get(self._akey(a), 0) == 0
        ]
        if no_visitadas:
            return self._rng.choice(no_visitadas)

        # UCB1: argmax Q(s,a) + c * sqrt(ln(N_s) / N_sa)
        mejor_accion = None
        mejor_puntaje = float("-inf")
        for a in acciones_disponibles:
            akey = self._akey(a)
            n_sa = self._N_sa[skey][akey]
            q = self._Q[skey][akey]
            bonus = self.c * math.sqrt(math.log(N_s) / n_sa)
            puntaje = q + bonus
            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_accion = a

        return mejor_accion

    def update(self, cache, requested_page, action, reward):
        """Actualiza el Q-valor y los contadores tras observar una recompensa.

        Usa la actualización incremental de la media:
            Q(s,a) <- Q(s,a) + (recompensa - Q(s,a)) / N_sa
        """
        skey = self._skey(cache, requested_page)
        akey = self._akey(action)

        if skey not in self._Q:
            self._Q[skey] = {}
            self._N_sa[skey] = {}
            self._N_s[skey] = 0

        if akey not in self._Q[skey]:
            self._Q[skey][akey] = 0.0
            self._N_sa[skey][akey] = 0

        self._N_s[skey] += 1
        self._N_sa[skey][akey] += 1
        n = self._N_sa[skey][akey]
        self._Q[skey][akey] += (reward - self._Q[skey][akey]) / n

    def reset(self):
        """Borra todas las estadísticas aprendidas."""
        self._Q = {}
        self._N_sa = {}
        self._N_s = {}

    def get_learned_policy(self):
        """Extrae la política greedy a partir de los Q-valores aprendidos.

        Retorna dict {(frozenset_cache, pagina_solicitada): mejor_accion}.
        Solo incluye estados que han sido visitados al menos una vez.
        """
        policy = {}
        for skey, actions in self._Q.items():
            if not actions:
                continue
            mejor_a = max(actions, key=lambda a: actions[a])
            policy[skey] = mejor_a
        return policy

    def stats_summary(self):
        """Imprime cuántos pares distintos (estado, acción) han sido explorados."""
        total_sa = sum(len(v) for v in self._N_sa.values())
        print(f"Estadísticas UCB: {len(self._N_s)} estados visitados, "
              f"{total_sa} pares (estado, acción) explorados.")