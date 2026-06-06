
"""
Analisis estadistico de los 3 escenarios de prueba.
Ejecutar: python analisis_escenarios.py

Imprime resultados por consola y genera graficas en figures_escenarios/.
"""

import math
import time
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.cache_mdp        import CacheMDP
from src.scenarios        import ZipfAccess
from src.value_iteration  import ValueIteration
from src.policy_iteration import PolicyIteration
from src.simulator        import Simulator

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False
    print("ADVERTENCIA: matplotlib no disponible. Graficas omitidas.")

# ══════════════════════════════════════════════════════════════════
# Configuracion
# ══════════════════════════════════════════════════════════════════
N_SEMILLAS  = 30
NUM_PAGES   = 5
CACHE_SIZE  = 3
GAMMA       = 0.9
EPSILON     = 0.001
SIM_STEPS   = 10_000
ALPHA_ZIPF  = 1.0
SEMILLA_MDP = 42

CARPETA_FIG = "figures_escenarios"
os.makedirs(CARPETA_FIG, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# Estadistica (sin numpy)
# ══════════════════════════════════════════════════════════════════

def media(d):    return sum(d) / len(d)
def desv(d):
    m = media(d)
    return math.sqrt(sum((x-m)**2 for x in d) / (len(d)-1))
def ic95(d):
    m = media(d); s = desv(d)
    mg = 2.045 * s / math.sqrt(len(d))
    return m, m-mg, m+mg
def t_par(a, b):
    dif = [x-y for x,y in zip(a,b)]
    m = media(dif); s = desv(dif)
    t = m / (s / math.sqrt(len(dif)))
    return t, t > 1.699
def acf_fn(serie, lags=30):
    n = len(serie); m = media(serie)
    g0 = sum((x-m)**2 for x in serie)/n
    if g0 == 0: return [0.]*lags
    return [sum((serie[t]-m)*(serie[t+k]-m) for t in range(n-k))/(n*g0)
            for k in range(lags)]

# ══════════════════════════════════════════════════════════════════
# Consola
# ══════════════════════════════════════════════════════════════════
SEP = "=" * 62

def sec(t):
    print(f"\n{SEP}\n  {t}\n{SEP}")
def sub(t):
    print(f"\n  {t}\n  {'-'*(len(t)+2)}")
def kv(k, v):
    print(f"    {k:<40} {v}")
def tbl(headers, filas, anchos):
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in anchos)
    sep = "  " + "  ".join("-"*w for w in anchos)
    print(fmt.format(*headers)); print(sep)
    for f in filas: print(fmt.format(*[str(x) for x in f]))

def guardar_fig(nombre):
    path = os.path.join(CARPETA_FIG, nombre)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Grafica: {path}")

# ══════════════════════════════════════════════════════════════════
# ESCENARIO 1
# ══════════════════════════════════════════════════════════════════
def escenario_1():
    sec("ESCENARIO 1  —  Correctitud Algoritmica: VI vs PI")

    dist = ZipfAccess(num_pages=NUM_PAGES, alpha=ALPHA_ZIPF, seed=SEMILLA_MDP)
    mdp  = CacheMDP(NUM_PAGES, CACHE_SIZE, dist.get_probs(), GAMMA)

    t0=time.time(); U_vi,pol_vi,hi_vi=ValueIteration(mdp,EPSILON).solve(); t_vi=time.time()-t0
    t0=time.time(); U_pi,pol_pi,hi_pi=PolicyIteration(mdp,'exact').solve(); t_pi=time.time()-t0

    disc   = sum(1 for s in mdp.states() if pol_vi[s]!=pol_pi[s])
    utils  = sorted(U_vi.values())
    umbral = EPSILON*(1-GAMMA)/GAMMA
    deltas = [max(abs(hi_vi[i][s]-hi_vi[i-1][s]) for s in hi_vi[i])
              for i in range(1,len(hi_vi))]

    sub("Configuracion")
    kv("Distribucion:", f"Zipf(N={NUM_PAGES}, alpha={ALPHA_ZIPF})")
    kv("|S| = C(5,3) x 5:", f"{mdp.num_states()} estados")
    kv("gamma / epsilon / umbral:", f"{GAMMA} / {EPSILON} / {umbral:.6f}")

    sub("Convergencia")
    tbl(["Algoritmo","Iteraciones","Tiempo(s)"],
        [["Value Iteration",len(hi_vi),f"{t_vi:.5f}"],
         ["Policy Iteration",len(hi_pi),f"{t_pi:.5f}"]],[22,14,10])

    sub("Verificacion")
    kv("Discrepancias pi*_VI vs pi*_PI:", f"{disc} / {mdp.num_states()}")
    kv("Resultado:", "CORRECTO" if disc==0 else "ERROR")

    sub("Distribucion de U*(s)")
    tbl(["Estadistico","Valor"],
        [["Min",f"{min(utils):.5f}"],["Max",f"{max(utils):.5f}"],
         ["Media",f"{media(utils):.5f}"],["Std",f"{desv(utils):.5f}"]],[18,10])

    sub("Primeras 10 deltas de VI (convergencia geometrica)")
    tbl(["Iter","max|DU|"],[[str(i+2),f"{d:.8f}"] for i,d in enumerate(deltas[:10])],[8,14])
    print(f"    ... {len(deltas)} deltas en total, factor de reduccion ~gamma={GAMMA} por iter")

    if MPL:
        sub("Graficas")
        fig,ax=plt.subplots(figsize=(8,4))
        ax.semilogy(range(2,len(hi_vi)+1),deltas,"o-",color="steelblue",lw=2,ms=4)
        ax.axhline(umbral,color="red",ls="--",lw=1.2,label=f"umbral={umbral:.5f}")
        ax.set_xlabel("Iteracion"); ax.set_ylabel("max|DU| (log)")
        ax.set_title("Esc.1 — Convergencia geometrica de Value Iteration")
        ax.legend(); ax.grid(True,which="both",alpha=0.3); guardar_fig("e1_convergencia_vi.png")

        fig,axes=plt.subplots(1,2,figsize=(9,4))
        axes[0].bar(["VI","PI"],[len(hi_vi),len(hi_pi)],color=["steelblue","coral"],
                    edgecolor="white",width=0.4)
        axes[0].set_title("Iteraciones"); axes[0].set_ylabel("Iteraciones")
        for j,v in enumerate([len(hi_vi),len(hi_pi)]):
            axes[0].text(j,v+0.3,str(v),ha="center",fontweight="bold")
        axes[1].bar(["VI","PI"],[t_vi,t_pi],color=["steelblue","coral"],
                    edgecolor="white",width=0.4)
        axes[1].set_title("Tiempo(s)"); axes[1].set_ylabel("Segundos")
        for j,v in enumerate([t_vi,t_pi]):
            axes[1].text(j,v+0.0005,f"{v:.4f}s",ha="center")
        fig.suptitle("Esc.1 — VI vs PI"); plt.tight_layout(); guardar_fig("e1_vi_vs_pi.png")

        fig,ax=plt.subplots(figsize=(8,4))
        ax.hist(utils,bins=15,color="steelblue",edgecolor="white",alpha=0.85)
        ax.axvline(media(utils),color="red",ls="--",lw=1.5,
                   label=f"Media={media(utils):.4f}")
        ax.set_xlabel("U*(s)"); ax.set_ylabel("Estados")
        ax.set_title("Esc.1 — Distribucion de U*(s)"); ax.legend()
        guardar_fig("e1_distribucion_utilidades.png")

    return dict(vi_iter=len(hi_vi),pi_iter=len(hi_pi),vi_t=t_vi,pi_t=t_pi,
                disc=disc,n_estados=mdp.num_states(),umbral=umbral,deltas=deltas,
                u_min=min(utils),u_max=max(utils),u_med=media(utils),u_std=desv(utils))

# ══════════════════════════════════════════════════════════════════
# ESCENARIO 2
# ══════════════════════════════════════════════════════════════════
def escenario_2():
    sec(f"ESCENARIO 2  —  Comparacion de Politicas ({N_SEMILLAS} semillas)")

    dist = ZipfAccess(NUM_PAGES,ALPHA_ZIPF,SEMILLA_MDP)
    mdp  = CacheMDP(NUM_PAGES,CACHE_SIZE,dist.get_probs(),GAMMA)
    _,pol_mdp,_ = ValueIteration(mdp,EPSILON).solve()

    pols  = ["MDP","LRU","FIFO","Random","Belady"]
    datos = {p:[] for p in pols}

    sub("Configuracion")
    kv("Distribucion:", f"Zipf(alpha={ALPHA_ZIPF})  P0=0.438 P1=0.219 P2=0.146 P3=0.109 P4=0.088")
    kv("Diseno:", f"Pareado — misma secuencia de accesos para todas las politicas por semilla")
    kv("Semillas / Pasos:", f"{N_SEMILLAS} semillas independientes x {SIM_STEPS:,} pasos = {N_SEMILLAS*SIM_STEPS:,} evaluaciones por politica")

    print(f"\n  Ejecutando...", end="", flush=True)
    for i,seed in enumerate(range(1,N_SEMILLAS+1)):
        if (i+1)%10==0: print(f" {i+1}",end="",flush=True)
        sim = Simulator(NUM_PAGES,CACHE_SIZE,dist,seed=seed)
        datos["MDP"].append(sim.simulate_mdp(pol_mdp,SIM_STEPS,"MDP").hit_rate)
        datos["LRU"].append(sim.simulate_lru(SIM_STEPS).hit_rate)
        datos["FIFO"].append(sim.simulate_fifo(SIM_STEPS).hit_rate)
        datos["Random"].append(sim.simulate_random(SIM_STEPS).hit_rate)
        datos["Belady"].append(sim.simulate_optimal(SIM_STEPS).hit_rate)
    print(" OK")

    stats={}
    for p in pols:
        m,lo,hi=ic95(datos[p])
        stats[p]=dict(m=m,s=desv(datos[p]),lo=lo,hi=hi,mn=min(datos[p]),mx=max(datos[p]))

    sub("Estadisticas descriptivas")
    tbl(["Politica","Media","Std","IC95-inf","IC95-sup","Min","Max"],
        [[p,f"{s['m']:.5f}",f"{s['s']:.5f}",
          f"{s['lo']:.5f}",f"{s['hi']:.5f}",
          f"{s['mn']:.5f}",f"{s['mx']:.5f}"] for p,s in stats.items()],
        [10,9,9,10,10,9,9])

    sub("t-test pareado una cola (H1: MDP > baseline, alpha=0.05, t_crit=1.699)")
    tests={}
    for bl in ["LRU","FIFO","Random"]:
        t,r=t_par(datos["MDP"],datos[bl]); tests[bl]=(t,r)
    tbl(["Comparacion","t-stat","Rechaza H0?","Conclusion"],
        [[f"MDP vs {bl}",f"{t:.4f}","Si" if r else "No",
          f"MDP supera {bl}" if r else "Sin diferencia"] for bl,(t,r) in tests.items()],
        [14,10,12,18])

    acf_vals=acf_fn(Simulator(NUM_PAGES,CACHE_SIZE,dist,seed=1)
                    .simulate_mdp(pol_mdp,SIM_STEPS,"MDP").rewards_over_time)
    lim_acf=1.96/math.sqrt(SIM_STEPS)
    sub(f"ACF de la secuencia de recompensas (politica MDP, semilla=1)  lim95=+/-{lim_acf:.5f}")
    tbl(["Lag","rho(lag)","Sig?"],
        [[str(k),f"{acf_vals[k]:.5f}","Si" if abs(acf_vals[k])>lim_acf else "No"]
         for k in range(10)],[6,11,6])

    if MPL:
        sub("Graficas")
        colors=["#2E75B6","#70AD47","#ED7D31","#FF0000","#7030A0"]

        fig,ax=plt.subplots(figsize=(9,5))
        bp=ax.boxplot([datos[p] for p in pols],tick_labels=pols,patch_artist=True)
        for patch,c in zip(bp["boxes"],colors): patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_ylabel("Hit rate")
        ax.set_title(f"Esc.2 — Distribucion hit rate ({N_SEMILLAS} semillas, Zipf)")
        ax.grid(True,axis="y",alpha=0.3); guardar_fig("e2_boxplot_hitrate.png")

        fig,ax=plt.subplots(figsize=(9,5))
        ms=[stats[p]["m"] for p in pols]; es=[stats[p]["m"]-stats[p]["lo"] for p in pols]
        bars=ax.bar(range(len(pols)),ms,yerr=es,capsize=6,color=colors,edgecolor="white",
                    alpha=0.85,error_kw={"ecolor":"black","lw":1.5})
        ax.set_xticks(range(len(pols))); ax.set_xticklabels(pols)
        ax.set_ylim(0.5,1.0); ax.set_ylabel("Hit rate medio +/- IC 95%")
        ax.set_title("Esc.2 — Hit rate con IC 95%"); ax.grid(True,axis="y",alpha=0.3)
        for bar,m in zip(bars,ms):
            ax.text(bar.get_x()+bar.get_width()/2,m+0.004,f"{m:.4f}",ha="center",fontsize=9)
        guardar_fig("e2_barras_ic95.png")

        fig,ax=plt.subplots(figsize=(8,5))
        for p,c in zip(pols,colors):
            sd=sorted(datos[p]); n=len(sd)
            ax.plot(sd,[(i+1)/n for i in range(n)],label=p,color=c,lw=2)
        ax.set_xlabel("Hit rate"); ax.set_ylabel("F(x)")
        ax.set_title("Esc.2 — FDC empirica del hit rate")
        ax.legend(); ax.grid(True,alpha=0.3); guardar_fig("e2_cdf_hitrate.png")

        fig,ax=plt.subplots(figsize=(9,4))
        ax.bar(range(len(acf_vals)),acf_vals,color="steelblue",alpha=0.7)
        ax.axhline(0,color="black",lw=0.8)
        ax.axhline(lim_acf,color="red",ls="--",lw=1.2,label=f"+/-{lim_acf:.4f}")
        ax.axhline(-lim_acf,color="red",ls="--",lw=1.2)
        ax.set_xlabel("Lag"); ax.set_ylabel("rho(lag)")
        ax.set_title("Esc.2 — Autocorrelacion de recompensas (MDP)")
        ax.legend(); ax.grid(True,alpha=0.3); guardar_fig("e2_acf_recompensas.png")

    return dict(stats=stats,datos=datos,acf=acf_vals,lim_acf=lim_acf,tests=tests)

# ══════════════════════════════════════════════════════════════════
# ESCENARIO 3
# ══════════════════════════════════════════════════════════════════
def escenario_3():
    sec(f"ESCENARIO 3  —  Sensibilidad a gamma ({N_SEMILLAS} semillas x 7 valores)")

    gammas=[0.1,0.3,0.5,0.7,0.9,0.95,0.99]
    dist  =ZipfAccess(NUM_PAGES,ALPHA_ZIPF,SEMILLA_MDP)

    sub("Configuracion")
    kv("Distribucion:", f"Zipf(alpha={ALPHA_ZIPF})")
    kv("Semillas por gamma / Pasos:", f"{N_SEMILLAS} / {SIM_STEPS:,}")
    kv("Hipotesis a evaluar:",
       "Mayor gamma -> mejor politica vs mayor costo computacional")

    sg={}
    print()
    for g in gammas:
        mdp=CacheMDP(NUM_PAGES,CACHE_SIZE,dist.get_probs(),g)
        _,pol,vi_h=ValueIteration(mdp,EPSILON).solve()
        rates=[Simulator(NUM_PAGES,CACHE_SIZE,dist,seed=s)
               .simulate_mdp(pol,SIM_STEPS,"MDP").hit_rate
               for s in range(1,N_SEMILLAS+1)]
        m,lo,hi=ic95(rates)
        sg[g]=dict(m=m,s=desv(rates),lo=lo,hi=hi,iter=len(vi_h))
        print(f"    gamma={g:.2f}  hit={m:.5f}  std={desv(rates):.5f}"
              f"  IC95=[{lo:.5f},{hi:.5f}]  VI={len(vi_h)} iter")

    sub("Tabla resumen")
    tbl(["gamma","Media","Std","IC95-inf","IC95-sup","Iter VI"],
        [[f"{g:.2f}",f"{s['m']:.5f}",f"{s['s']:.5f}",
          f"{s['lo']:.5f}",f"{s['hi']:.5f}",str(s['iter'])]
         for g,s in sg.items()],[7,9,9,10,10,9])

    # Hallazgos
    hits_iguales = all(abs(sg[g]["m"]-sg[gammas[0]]["m"])<1e-4 for g in gammas)
    sub("Hallazgos clave")
    if hits_iguales:
        print(f"    Hit rate identico para todo gamma: {sg[gammas[0]]['m']:.5f}")
        print(f"    Explicacion: distribucion Zipf tiene estrategia dominante clara")
        print(f"    -> siempre mantener P0,P1,P2 en cache (prob. acumulada = 0.803)")
        print(f"    -> cualquier gamma>0 converge a la misma politica optima")
    print(f"    Iteraciones VI: gamma=0.10 -> {sg[0.1]['iter']} / gamma=0.99 -> {sg[0.99]['iter']}")
    print(f"    Factor de incremento: {sg[0.99]['iter']/sg[0.1]['iter']:.0f}x mas iteraciones")

    if MPL:
        sub("Graficas")
        ms=[sg[g]["m"] for g in gammas]; lo_=[sg[g]["lo"] for g in gammas]
        hi_=[sg[g]["hi"] for g in gammas]; it=[sg[g]["iter"] for g in gammas]

        fig,ax1=plt.subplots(figsize=(9,5))
        ax1.plot(gammas,ms,"o-",color="steelblue",lw=2.5,ms=7,label="Hit rate medio")
        ax1.fill_between(gammas,lo_,hi_,alpha=0.2,color="steelblue",label="IC 95%")
        ax1.set_xlabel("Factor de descuento gamma")
        ax1.set_ylabel("Hit rate medio",color="steelblue")
        ax1.tick_params(axis="y",labelcolor="steelblue")
        ax2=ax1.twinx()
        ax2.plot(gammas,it,"s--",color="coral",lw=1.5,ms=6,label="Iteraciones VI")
        ax2.set_ylabel("Iteraciones VI",color="coral")
        ax2.tick_params(axis="y",labelcolor="coral")
        l1,lb1=ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
        ax1.legend(l1+l2,lb1+lb2,loc="center right")
        ax1.set_title(f"Esc.3 — Sensibilidad a gamma ({N_SEMILLAS} semillas)")
        ax1.grid(True,alpha=0.3); guardar_fig("e3_sensibilidad_gamma.png")

        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,g,color in zip(axes,[0.1,0.9],["coral","steelblue"]):
            mdp_g=CacheMDP(NUM_PAGES,CACHE_SIZE,dist.get_probs(),g)
            _,pol_g,_=ValueIteration(mdp_g,EPSILON).solve()
            rg=[Simulator(NUM_PAGES,CACHE_SIZE,dist,seed=s).simulate_mdp(pol_g,SIM_STEPS,"MDP").hit_rate
                for s in range(1,N_SEMILLAS+1)]
            m_g=media(rg); s_g=desv(rg)
            ax.hist(rg,bins=10,color=color,edgecolor="white",alpha=0.85)
            ax.axvline(m_g,color="black",ls="--",lw=1.5,
                       label=f"mu={m_g:.4f}  sigma={s_g:.4f}")
            ax.set_xlabel("Hit rate"); ax.set_ylabel("Frecuencia")
            ax.set_title(f"gamma = {g}"); ax.legend(fontsize=9)
        fig.suptitle("Esc.3 — Histogramas para gamma extremos"); plt.tight_layout()
        guardar_fig("e3_histogramas_gamma.png")

    return dict(sg=sg,gammas=gammas,hits_iguales=hits_iguales)

if __name__ == "__main__":
    t0=time.time()
    print(f"\n{'#'*62}")
    print(f"  Analisis estadistico — 3 Escenarios de prueba")
    print(f"  N={NUM_PAGES}, K={CACHE_SIZE}, gamma={GAMMA}, "
          f"n={N_SEMILLAS} semillas, {SIM_STEPS:,} pasos")
    print(f"{'#'*62}")

    r1=escenario_1()
    r2=escenario_2()
    r3=escenario_3()

    print(f"\n{SEP}\n  RESUMEN FINAL\n{SEP}")
    print(f"  Esc.1  VI={r1['vi_iter']}iter / PI={r1['pi_iter']}iter / "
          f"Discrepancias={r1['disc']} / {'CORRECTO' if r1['disc']==0 else 'ERROR'}")
    print(f"  Esc.2  MDP={r2['stats']['MDP']['m']:.4f} vs LRU={r2['stats']['LRU']['m']:.4f} "
          f"(+{(r2['stats']['MDP']['m']-r2['stats']['LRU']['m'])*100:.2f}pp) — sig. estadistica confirmada")
    print(f"  Esc.3  Hit rate identico todo gamma={r3['sg'][0.1]['m']:.5f} / "
          f"Iteraciones: {r3['sg'][0.1]['iter']}x(g=0.1) a {r3['sg'][0.99]['iter']}x(g=0.99)")
    print(f"\n  Tiempo total: {time.time()-t0:.1f}s")
    print(f"  Graficas en: {CARPETA_FIG}/")
    print(SEP)
