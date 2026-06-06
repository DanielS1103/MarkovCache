
"""
Distribuciones de probabilidad de acceso a páginas para los experimentos de simulación.

Todas las distribuciones implementan:
  get_probs() -> dict {page_id: probabilidad}  (suma 1.0)
  sample()    -> page_id                        (genera una solicitud)
  name()      -> str
"""

import random
import math


class AccessDistribution:
    """Clase base para patrones de acceso a páginas."""

    def __init__(self, num_pages, seed=None):
        self.num_pages = num_pages
        self.page_ids = list(range(num_pages))
        self._rng = random.Random(seed)
        self._probs = None  # calculado por la subclase

    def get_probs(self):
        """Retorna diccionario {page_id: probabilidad}."""
        raise NotImplementedError

    def sample(self):
        """Genera una solicitud de página según la distribución."""
        probs = self.get_probs()
        r = self._rng.random()
        cumulative = 0.0
        for page, p in sorted(probs.items()):
            cumulative += p
            if r <= cumulative:
                return page
        return self.page_ids[-1]  # fallback numérico

    def name(self):
        raise NotImplementedError

    def __repr__(self):
        return self.name()


# ------------------------------------------------------------------
# Distribución uniforme
# ------------------------------------------------------------------

class UniformAccess(AccessDistribution):
    """Todas las páginas con la misma probabilidad: prob = 1/N."""

    def get_probs(self):
        if self._probs is None:
            p = 1.0 / self.num_pages
            self._probs = {pg: p for pg in self.page_ids}
        return self._probs

    def name(self):
        return f"Uniforme(N={self.num_pages})"


# ------------------------------------------------------------------
# Distribución Zipf
# ------------------------------------------------------------------

class ZipfAccess(AccessDistribution):
    """Ley de Zipf: prob(rango r) proporcional a 1/r^alpha.

    Las páginas se ordenan de mayor a menor popularidad (page_id=0 es la más popular).
    alpha=1.0 es la Zipf clásica. Mayor alpha implica mayor sesgo.
    """

    def __init__(self, num_pages, alpha=1.0, seed=None):
        super().__init__(num_pages, seed)
        self.alpha = alpha

    def get_probs(self):
        if self._probs is None:
            rangos = [i + 1 for i in range(self.num_pages)]  # rangos 1..N
            pesos = [1.0 / (r ** self.alpha) for r in rangos]
            total = sum(pesos)
            self._probs = {
                pg: pesos[pg] / total
                for pg in self.page_ids
            }
        return self._probs

    def name(self):
        return f"Zipf(N={self.num_pages}, alpha={self.alpha})"


# ------------------------------------------------------------------
# Distribución de localidad temporal
# ------------------------------------------------------------------

class TemporalLocalityAccess(AccessDistribution):
    """Modelo de conjunto de trabajo: un subconjunto de páginas 'calientes' concentra el tráfico.

    Args:
        num_hot_pages: número de páginas en el conjunto de trabajo activo
        hot_prob:      masa de probabilidad total asignada a las páginas calientes
    """

    def __init__(self, num_pages, num_hot_pages=None, hot_prob=0.8, seed=None):
        super().__init__(num_pages, seed)
        self.num_hot = num_hot_pages if num_hot_pages is not None else max(1, num_pages // 3)
        self.hot_prob = hot_prob
        assert self.num_hot < num_pages, "num_hot_pages debe ser < num_pages"
        assert 0 < hot_prob < 1, "hot_prob debe estar en (0,1)"

    def get_probs(self):
        if self._probs is None:
            hot_pages = self.page_ids[: self.num_hot]
            cold_pages = self.page_ids[self.num_hot :]

            hot_each = self.hot_prob / self.num_hot
            cold_each = (1.0 - self.hot_prob) / len(cold_pages) if cold_pages else 0.0

            self._probs = {}
            for pg in hot_pages:
                self._probs[pg] = hot_each
            for pg in cold_pages:
                self._probs[pg] = cold_each
        return self._probs

    def name(self):
        return f"LocalidadTemporal(N={self.num_pages}, hot={self.num_hot}, p={self.hot_prob})"


# ------------------------------------------------------------------
# Distribución no estacionaria (cambia en un paso dado)
# ------------------------------------------------------------------

class NonStationaryAccess:
    """Distribución que cambia en un paso de simulación especificado.

    Útil para demostrar que UCB puede adaptarse mientras que una política
    MDP fija no puede responder al cambio.
    """

    def __init__(self, dist1, dist2, switch_step, seed=None):
        self.dist1 = dist1
        self.dist2 = dist2
        self.switch_step = switch_step
        self._rng = random.Random(seed)
        self._step = 0

    def get_probs(self, step=None):
        s = step if step is not None else self._step
        return self.dist1.get_probs() if s < self.switch_step else self.dist2.get_probs()

    def sample(self):
        probs = self.get_probs(self._step)
        self._step += 1
        r = self._rng.random()
        cumulative = 0.0
        for page, p in sorted(probs.items()):
            cumulative += p
            if r <= cumulative:
                return page
        return sorted(probs.keys())[-1]

    def reset(self):
        self._step = 0

    def name(self):
        return (f"NoEstacionaria(cambio@{self.switch_step}: "
                f"{self.dist1.name()} -> {self.dist2.name()})")

    def __repr__(self):
        return self.name()


# ------------------------------------------------------------------
# Función auxiliar para imprimir un resumen de la distribución
# ------------------------------------------------------------------

def print_distribution(dist):
    probs = dist.get_probs()
    print(f"\nDistribución: {dist.name()}")
    print(f"{'Página':<8} {'Prob':>10}")
    print("-" * 20)
    for pg in sorted(probs.keys()):
        barra = "#" * int(probs[pg] * 40)
        print(f"P{pg:<7} {probs[pg]:>10.4f}  {barra}")