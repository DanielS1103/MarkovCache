
"""
Políticas clásicas de reemplazo de caché usadas como baseline de comparación.

Todas las políticas implementan:
    decide_eviction(cache: frozenset, requested_page: int) -> página_a_desalojar

  - LRUPolicy:             Least Recently Used (menos recientemente usado)
  - FIFOPolicy:            First In, First Out (primero en entrar, primero en salir)
  - RandomPolicy:          Desalojar una página al azar
  - OptimalOfflinePolicy:  Algoritmo de Belady (requiere conocer el futuro)
"""

import random


class LRUPolicy:
    """LRU — desaloja la página que fue accedida hace más tiempo.

    Mantiene una lista ordenada de páginas por recencia de acceso.
    """

    def __init__(self, seed=None):
        self._access_order = []   # la más reciente al final
        self._rng = random.Random(seed)

    def on_access(self, page):
        """Registra que 'page' acaba de ser accedida (llamar en cada hit Y miss)."""
        if page in self._access_order:
            self._access_order.remove(page)
        self._access_order.append(page)

    def decide_eviction(self, cache, requested_page):
        """Desaloja la página en caché que fue accedida hace más tiempo."""
        for page in self._access_order:
            if page in cache:
                return page
        # Fallback: desalojar cualquier página (no debería ocurrir en operación normal)
        return next(iter(cache))

    def reset(self):
        self._access_order = []

    def name(self):
        return "LRU"


class FIFOPolicy:
    """FIFO — desaloja la página que lleva más tiempo en caché."""

    def __init__(self, seed=None):
        self._insertion_order = []   # la más antigua al frente

    def on_load(self, page):
        """Registra que 'page' acaba de ser cargada en caché."""
        if page not in self._insertion_order:
            self._insertion_order.append(page)

    def decide_eviction(self, cache, requested_page):
        """Desaloja la página en caché que fue cargada primero."""
        for page in self._insertion_order:
            if page in cache:
                return page
        return next(iter(cache))

    def reset(self):
        self._insertion_order = []

    def name(self):
        return "FIFO"


class RandomPolicy:
    """Random — desaloja una página uniformemente aleatoria del caché."""

    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def decide_eviction(self, cache, requested_page):
        return self._rng.choice(sorted(cache))

    def reset(self):
        pass

    def name(self):
        return "Random"


class OptimalOfflinePolicy:
    """Algoritmo óptimo offline de Belady: desaloja la página cuyo próximo uso es más lejano.

    Requiere conocer la secuencia completa de accesos de antemano (no es realista),
    pero provee una cota superior teórica para la comparación.
    """

    def __init__(self, access_sequence):
        self._sequence = access_sequence
        self._current_step = 0

    def decide_eviction(self, cache, requested_page):
        """Desaloja la página en caché cuyo próximo acceso está más lejos en el futuro."""
        futuro = self._sequence[self._current_step:]
        pagina_mas_lejana = None
        distancia_max = -1

        for page in cache:
            # Buscar el próximo acceso de esta página en la secuencia futura
            try:
                dist = futuro.index(page)
            except ValueError:
                # La página nunca más será accedida: desalojar de inmediato
                return page
            if dist > distancia_max:
                distancia_max = dist
                pagina_mas_lejana = page

        return pagina_mas_lejana

    def on_step(self):
        self._current_step += 1

    def reset(self):
        self._current_step = 0

    def name(self):
        return "Óptimo (Belady)"