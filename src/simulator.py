
"""
Motor de simulación: ejecuta el caché con distintas políticas y recolecta métricas.

Soporta:
  - Política MDP (diccionario pre-calculado con Value/Policy Iteration)
  - Políticas baseline clásicas (LRU, FIFO, Random)
  - Agente UCB online
  - Óptimo offline (Belady)
"""

import random


class SimResult:
    """Almacena los resultados de una corrida de simulación."""

    def __init__(self, policy_name, num_steps):
        self.policy_name = policy_name
        self.num_steps = num_steps
        self.hits = 0
        self.misses = 0
        self.rewards_over_time = []       # +1 o -1 por paso
        self.cumulative_rewards = []      # suma acumulada
        self._cumulative = 0.0

    def record(self, hit):
        reward = 1.0 if hit else -1.0
        self.rewards_over_time.append(reward)
        self._cumulative += reward
        self.cumulative_rewards.append(self._cumulative)
        if hit:
            self.hits += 1
        else:
            self.misses += 1

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_reward(self):
        return float(self.hits - self.misses)

    def summary(self):
        return (f"{self.policy_name:<20} | "
                f"hits={self.hits:>6}, misses={self.misses:>6}, "
                f"hit_rate={self.hit_rate:.4f}, "
                f"recompensa_total={self.total_reward:>8.1f}")


class Simulator:
    """Simula el caché a lo largo de una secuencia de solicitudes de páginas.

    Args:
        num_pages:    número total de páginas distintas (N)
        cache_size:   capacidad del caché (K)
        access_dist:  instancia de AccessDistribution
        seed:         semilla para secuencias de acceso reproducibles
    """

    def __init__(self, num_pages, cache_size, access_dist, seed=42):
        self.num_pages = num_pages
        self.cache_size = cache_size
        self.access_dist = access_dist
        self._seed = seed

    def _generate_sequence(self, num_steps):
        """Genera la secuencia de solicitudes de páginas según la distribución de acceso."""
        rng = random.Random(self._seed)
        probs = self.access_dist.get_probs()
        pages = sorted(probs.keys())
        weights = [probs[p] for p in pages]
        sequence = []
        for _ in range(num_steps):
            r = rng.random()
            cumulative = 0.0
            chosen = pages[-1]
            for p, w in zip(pages, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = p
                    break
            sequence.append(chosen)
        return sequence

    # ------------------------------------------------------------------
    # Simulación con política MDP
    # ------------------------------------------------------------------

    def simulate_mdp(self, policy, num_steps=10000, policy_name="MDP"):
        """Simula usando una política MDP pre-calculada.

        El diccionario 'policy' mapea (frozenset(cache), pagina_solicitada) -> acción.
        """
        result = SimResult(policy_name, num_steps)
        sequence = self._generate_sequence(num_steps)

        # Iniciar con caché vacío; llenarlo con las primeras K páginas distintas
        cache = set()
        for page in sequence:
            if page not in cache:
                cache.add(page)
            if len(cache) >= self.cache_size:
                break

        cache = frozenset(cache)

        for page in sequence:
            state = (cache, page)
            action = policy.get(state)

            if page in cache:
                # Hit: no se necesita desalojo
                result.record(hit=True)
            else:
                # Miss
                result.record(hit=False)
                if len(cache) < self.cache_size:
                    # Aún hay espacio: simplemente cargar
                    cache = cache | {page}
                elif action is not None and action != "noop":
                    _, evicted = action
                    cache = (cache - {evicted}) | {page}
                else:
                    # Fallback: desalojar página arbitraria (estado no cubierto por la política)
                    evicted = next(iter(sorted(cache)))
                    cache = (cache - {evicted}) | {page}

        return result

    # ------------------------------------------------------------------
    # Simulación LRU
    # ------------------------------------------------------------------

    def simulate_lru(self, num_steps=10000):
        from src.baselines import LRUPolicy
        policy = LRUPolicy(seed=self._seed)
        result = SimResult("LRU", num_steps)
        sequence = self._generate_sequence(num_steps)

        cache = set()

        for page in sequence:
            if page in cache:
                result.record(hit=True)
                policy.on_access(page)
            else:
                result.record(hit=False)
                policy.on_access(page)
                if len(cache) < self.cache_size:
                    cache.add(page)
                else:
                    evicted = policy.decide_eviction(frozenset(cache), page)
                    cache.discard(evicted)
                    cache.add(page)

        return result

    # ------------------------------------------------------------------
    # Simulación FIFO
    # ------------------------------------------------------------------

    def simulate_fifo(self, num_steps=10000):
        from src.baselines import FIFOPolicy
        policy = FIFOPolicy()
        result = SimResult("FIFO", num_steps)
        sequence = self._generate_sequence(num_steps)

        cache = set()

        for page in sequence:
            if page in cache:
                result.record(hit=True)
            else:
                result.record(hit=False)
                if len(cache) < self.cache_size:
                    cache.add(page)
                    policy.on_load(page)
                else:
                    evicted = policy.decide_eviction(frozenset(cache), page)
                    cache.discard(evicted)
                    cache.add(page)
                    policy.on_load(page)

        return result

    # ------------------------------------------------------------------
    # Simulación Random
    # ------------------------------------------------------------------

    def simulate_random(self, num_steps=10000):
        from src.baselines import RandomPolicy
        policy = RandomPolicy(seed=self._seed + 1)
        result = SimResult("Random", num_steps)
        sequence = self._generate_sequence(num_steps)

        cache = set()

        for page in sequence:
            if page in cache:
                result.record(hit=True)
            else:
                result.record(hit=False)
                if len(cache) < self.cache_size:
                    cache.add(page)
                else:
                    evicted = policy.decide_eviction(frozenset(cache), page)
                    cache.discard(evicted)
                    cache.add(page)

        return result

    # ------------------------------------------------------------------
    # Simulación UCB
    # ------------------------------------------------------------------

    def simulate_ucb(self, ucb_agent, num_steps=10000):
        result = SimResult("UCB", num_steps)
        sequence = self._generate_sequence(num_steps)

        cache = set()
        ucb_agent.reset()

        for page in sequence:
            action = ucb_agent.select_action(frozenset(cache), page)

            if page in cache:
                reward = 1.0
                result.record(hit=True)
            else:
                reward = -1.0
                result.record(hit=False)
                if len(cache) < self.cache_size:
                    cache.add(page)
                elif action != "noop":
                    _, evicted = action
                    cache.discard(evicted)
                    cache.add(page)

            ucb_agent.update(frozenset(cache), page, action, reward)

        return result

    # ------------------------------------------------------------------
    # Simulación Óptimo offline (Belady)
    # ------------------------------------------------------------------

    def simulate_optimal(self, num_steps=10000):
        from src.baselines import OptimalOfflinePolicy
        sequence = self._generate_sequence(num_steps)
        policy = OptimalOfflinePolicy(sequence)
        result = SimResult("Óptimo(Belady)", num_steps)

        cache = set()

        for i, page in enumerate(sequence):
            policy._current_step = i + 1  # el futuro comienza después del paso actual

            if page in cache:
                result.record(hit=True)
            else:
                result.record(hit=False)
                if len(cache) < self.cache_size:
                    cache.add(page)
                else:
                    evicted = policy.decide_eviction(frozenset(cache), page)
                    cache.discard(evicted)
                    cache.add(page)

        return result

    # ------------------------------------------------------------------
    # Ejecutar todas las políticas e imprimir tabla comparativa
    # ------------------------------------------------------------------

    def run_all(self, mdp_vi_policy=None, mdp_pi_policy=None,
                ucb_agent=None, num_steps=10000, include_optimal=False):
        """Ejecuta todas las políticas y retorna la lista de SimResult."""
        results = []

        if mdp_vi_policy is not None:
            results.append(self.simulate_mdp(mdp_vi_policy, num_steps, "MDP-VI"))
        if mdp_pi_policy is not None:
            results.append(self.simulate_mdp(mdp_pi_policy, num_steps, "MDP-PI"))

        results.append(self.simulate_lru(num_steps))
        results.append(self.simulate_fifo(num_steps))
        results.append(self.simulate_random(num_steps))

        if ucb_agent is not None:
            results.append(self.simulate_ucb(ucb_agent, num_steps))
        if include_optimal:
            results.append(self.simulate_optimal(num_steps))

        return results

    @staticmethod
    def print_results(results):
        """Imprime la tabla comparativa de resultados."""
        print(f"\n{'Política':<20} | {'Hits':>6}  {'Misses':>6}  {'Hit Rate':>9}  {'Recompensa':>12}")
        print("-" * 65)
        for r in results:
            print(r.summary())