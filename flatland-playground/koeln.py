# scenario_koeln.py
"""
Basis-Topologie "Hansaring ↔ Hbf ↔ Hohenzollernbrücke ↔ Messe/Deutz"
(Version 1 – parametrisiert mit sparse_rail_generator)

Ziele:
- Reproduzierbares Engstellen-Szenario mit getrenntem S-Bahn- und Fern/RE-Korridor (approximiert)
- Zugklassen (FV/RE/S/Güter) + einfache Kosten-/Reward-Gewichte
- CLI-Flags ähnlich zu tester.py; kann standalone laufen

Hinweis:
Diese v1 bildet die Brücken-6-Gleis-Realität funktional ab (4 "main" + 2 "S-Bahn" als Kapazitätsidee),
aber nutzt noch keinen handgezeichneten Grid-Plan. Das folgt als v2 (handcrafted GridTransitions),
falls wir 6 exakte Parallelgleise als Zellen abbilden wollen.
"""
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.utils.rendertools import RenderTool


# -----------------------------
# ÖKONOMIE / KOSTEN-GEWICHTE
# -----------------------------
@dataclass
class CostWeights:
    w_fv: float = 5.0      # Fernverkehr (ICE/IC)
    w_re: float = 2.0      # Regionalverkehr (RE/RB)
    w_s: float = 1.5       # S-Bahn
    w_g: float = 1.0       # Güter
    deadlock_penalty: float = 50.0
    goal_reward: float = 5.0


# Klassen-Typen
TRAIN_CLASSES = ["FV", "RE", "S", "G"]

def assign_classes(num_agents: int) -> List[str]:
    """Zuweisung der Klassen (zyklisch) für genau die Zahl Agents,
    die das Environment nach reset() tatsächlich liefert."""
    return [TRAIN_CLASSES[i % len(TRAIN_CLASSES)] for i in range(num_agents)]


def class_weight(train_class: str, cw: CostWeights) -> float:
    return {
        "FV": cw.w_fv,
        "RE": cw.w_re,
        "S": cw.w_s,
        "G": cw.w_g,
    }[train_class]


# -----------------------------
# ENV-SETUP (parametrisiert)
# -----------------------------

def build_env(width: int, height: int, agents: int, cities: int, rails_between: int,
              rail_pairs_in_city: int, seed: int) -> RailEnv:
    # sparse_rail_generator mit höherer Parallelisierung, um die Brücken-/Bahnhofskapazität zu approximieren
    rail_gen = sparse_rail_generator(
        max_num_cities=max(2, cities),
        grid_mode=False,  # freiere Platzierung (kannst du auf True setzen, wenn dir Raster besser gefällt)
        max_rails_between_cities=max(1, rails_between),    # \approx Anzahl paralleler Korridorgleise
        max_rail_pairs_in_city=max(1, rail_pairs_in_city), # \approx Bahnsteig-/Weichen-Kapazität
        seed=seed,
    )

    env = RailEnv(
        width=width,
        height=height,
        number_of_agents=agents,
        rail_generator=rail_gen,
        line_generator=sparse_line_generator(),
        obs_builder_object=GlobalObsForRailEnv(),
    )
    return env


# -----------------------------
# EINFACHER EPISODEN-LAUF MIT KOSTEN-LOGGING
# -----------------------------

def run_episode(env: RailEnv, 
                train_classes: List[str],
                costs: CostWeights,
                render: str = "pgl",
                screen_w: int = 900,
                screen_h: int = 900,
                steps: int = 300,
                fps: int = 30,
                seed: int = 7) -> Dict[str, float]:

    rng = np.random.default_rng(seed)
    obs, info = env.reset()

    # Renderer optional
    renderer = None
    gl_backend = None
    if render != "off":
        try:
            if render == "pgl":
                renderer = RenderTool(env, gl="PGL", screen_width=screen_w, screen_height=screen_h)
                gl_backend = "PGL"
            elif render == "png":
                renderer = RenderTool(env, gl="PILSVG", screen_width=screen_w, screen_height=screen_h)
                gl_backend = "PILSVG"
            else:
                renderer = None
        except Exception as e:
            print(f"[warn] Render init failed ({render}): {e}. Fallback to PILSVG")
            renderer = RenderTool(env, gl="PILSVG", screen_width=screen_w, screen_height=screen_h)
            gl_backend = "PILSVG"

    # Kosten-Tracking
    total_penalty = 0.0
    done_agents = set()

    for t in range(steps):
        actions = {}
        # Simple Baseline: MOVE_FORWARD, sonst DO_NOTHING (kein Deadlock-Fix; nur als Platzhalter)
        for a in range(env.get_num_agents()):
            # ganz simple Heuristik: 80% forward, 20% stop – vermeidet ständiges Vorrollen bei Block
            actions[a] = 2 if rng.random() < 0.8 else 0

        obs, rewards, done, info = env.step(actions)

        # Kosten / Reward (Basis):
        # - Jede negative Belohnung interpretieren wir als Verspätungszuwachs (approximiert)
        # - gewichtet nach Klassen
        for a, r in rewards.items():
            if a in done_agents:
                continue
            w = class_weight(train_classes[a], costs)
            total_penalty += -min(0.0, r) * w  # nur negative Rewards als "Kosten" zählen

        # Render
        if renderer is not None:
            renderer.render_env(show=(gl_backend == "PGL"), show_observations=False, show_predictions=False)

        # Done-Handling
        for a, d in done.items():
            if a == "__all__":
                continue
            if d and a not in done_agents:
                done_agents.add(a)
                total_penalty -= costs.goal_reward  # Zielbonus (verringert Gesamtkosten)

        if done.get("__all__", False):
            print(f"[info] Episode finished early at step {t+1}.")
            break

        if fps > 0:
            time.sleep(1.0 / fps)

    if renderer is not None and gl_backend == "PILSVG":
        renderer.render_env(show=False, show_observations=False, show_predictions=False)
        renderer.save_render(f"koeln_frame_seed{seed}.png")

    # Deadlock-Penalty (falls keiner fertig wurde, z. B. blockiert)
    if len(done_agents) == 0:
        total_penalty += costs.deadlock_penalty

    return {
        "total_cost": total_penalty,
        "done_agents": float(len(done_agents)),
        "agents": float(env.get_num_agents()),
    }


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Köln-Engstelle – Basis-Topologie & Kostenlauf")

    # Topologie
    p.add_argument("--width", type=int, default=60)
    p.add_argument("--height", type=int, default=30)
    p.add_argument("--agents", type=int, default=12)
    p.add_argument("--cities", type=int, default=3, help=">=2; 3≈(Hansaring/Hbf/Deutz) approximiert")
    p.add_argument("--rails-between", type=int, default=4, help="≈ Parallelgleise zwischen Knoten (Brücke main)")
    p.add_argument("--rail-pairs-in-city", type=int, default=4, help="≈ Bahnsteig-/Weichen-Kapazität pro Knoten")
    p.add_argument("--seed", type=int, default=42)

    # Rendering
    p.add_argument("--render", choices=["pgl", "png", "off"], default="pgl")
    p.add_argument("--screen-w", type=int, default=900)
    p.add_argument("--screen-h", type=int, default=900)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--fps", type=int, default=20)

    # Kosten-Gewichte
    p.add_argument("--w-fv", type=float, default=5.0)
    p.add_argument("--w-re", type=float, default=2.0)
    p.add_argument("--w-s", type=float, default=1.5)
    p.add_argument("--w-g", type=float, default=1.0)
    p.add_argument("--goal-reward", type=float, default=5.0)
    p.add_argument("--deadlock-penalty", type=float, default=50.0)

    return p.parse_args()


def main():
    args = parse_args()

    env = build_env(
        width=args.width,
        height=args.height,
        agents=args.agents,
        cities=args.cities,
        rails_between=args.rails_between,
        rail_pairs_in_city=args.rail_pairs_in_city,
        seed=args.seed,
    )

     # Resets mit Seed-Retry
    for attempt in range(5):
        try:
            obs, info = env.reset()
            n_agents = env.get_num_agents()
            tclasses = assign_classes(n_agents)
            print(f"[info] Env ready: Agents={n_agents} Seed={args.seed} Try={attempt+1}")
            break
        except ValueError as e:
            ...

    costs = CostWeights(
        w_fv=args.w_fv,
        w_re=args.w_re,
        w_s=args.w_s,
        w_g=args.w_g,
        goal_reward=args.goal_reward,
        deadlock_penalty=args.deadlock_penalty,
    )

    metrics = run_episode(env, tclasses, costs,
                          render=args.render,
                          screen_w=args.screen_w,
                          screen_h=args.screen_h,
                          steps=args.steps,
                          fps=args.fps,
                          seed=args.seed)

    print("\n[RESULT] total_cost={:.2f} done_agents={}/{}".format(
        metrics["total_cost"], int(metrics["done_agents"]), int(metrics["agents"]))
    )


if __name__ == "__main__":
    main()
