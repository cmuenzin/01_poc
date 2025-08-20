# tester.py
import argparse
import time
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv

# Optional: wir importieren RenderTool erst in run_episode(), damit "off" ohne pyglet läuft
RenderTool = None

def build_env(args):
    rail_gen = sparse_rail_generator(
        max_num_cities=max(2, args.cities),
        grid_mode=False,
        max_rails_between_cities=max(1, args.rails_between),
        max_rail_pairs_in_city=max(1, args.rail_pairs_in_city),
        seed=args.seed,
    )

    env = RailEnv(
        width=args.width,
        height=args.height,
        number_of_agents=args.agents,
        rail_generator=rail_gen,
        line_generator=sparse_line_generator(),
        obs_builder_object=GlobalObsForRailEnv(),
    )
    return env


def run_episode(env, args):
    global RenderTool
    obs, info = env.reset()

    renderer = None
    gl_backend = None

    if args.render != "off":
        from flatland.utils.rendertools import RenderTool as _RT
        RenderTool = _RT

        if args.render == "pgl":
            # Versuche Live-Fenster (OpenGL via pyglet)
            try:
                renderer = RenderTool(env, gl="PGL", screen_width=args.screen_w, screen_height=args.screen_h)
                gl_backend = "PGL"
            except Exception as e:
                print(f"[warn] PGL init failed: {e}\n        Falling back to PILSVG (offscreen).")
                renderer = RenderTool(env, gl="PILSVG", screen_width=args.screen_w, screen_height=args.screen_h)
                gl_backend = "PILSVG"
        elif args.render == "png":
            renderer = RenderTool(env, gl="PILSVG", screen_width=args.screen_w, screen_height=args.screen_h)
            gl_backend = "PILSVG"

    # Random policy (Smoke-Test)
    for t in range(args.steps):
        actions = {a: np.random.randint(0, 5) for a in range(env.get_num_agents())}
        obs, rewards, done, info = env.step(actions)

        if renderer is not None:
            # show=True öffnet/aktualisiert Fenster für PGL, bleibt offscreen für PILSVG
            renderer.render_env(show=(gl_backend == "PGL"),
                                show_observations=False,
                                show_predictions=False)

        if done.get("__all__", False):
            print(f"[info] Episode finished early at step {t+1}.")
            break

        if args.fps > 0:
            time.sleep(1.0 / args.fps)

    if renderer is not None and gl_backend == "PILSVG":
        # Bei offscreen-Rendering wenigstens ein Frame speichern
        renderer.render_env(show=False, show_observations=False, show_predictions=False)
        out_name = f"frame_seed{args.seed}_agents{env.get_num_agents()}.png"
        renderer.save_render(out_name)
        print(f"[info] Saved offscreen frame -> {out_name}")

    print("[ok] Done.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Flatland tester: reproducible smoke-tests with live/PNG rendering."
    )
    # Env shape
    p.add_argument("--width", type=int, default=30, help="Grid width")
    p.add_argument("--height", type=int, default=30, help="Grid height")
    p.add_argument("--agents", type=int, default=3, help="Number of trains (agents)")
    p.add_argument("--cities", type=int, default=2, help="Approx. number of cities (>=2 recommended)")
    p.add_argument("--rails-between", type=int, default=2, help="Max rails between cities")
    p.add_argument("--rail-pairs-in-city", type=int, default=2, help="Max rail pairs in a city")
    p.add_argument("--seed", type=int, default=7, help="Base seed for generators")

    # Rendering
    p.add_argument("--render", choices=["pgl", "png", "off"], default="pgl",
                   help="pgl=live window (pyglet/OpenGL), png=offscreen PNG via PILSVG, off=no rendering")
    p.add_argument("--screen-w", type=int, default=800, help="Render width")
    p.add_argument("--screen-h", type=int, default=800, help="Render height")
    p.add_argument("--fps", type=int, default=30, help="Target FPS for live stepping (0 = no sleep)")

    # Episode
    p.add_argument("--steps", type=int, default=200, help="Max steps per episode")

    return p.parse_args()


def main():
    args = parse_args()
    env = build_env(args)

    # Robust reset: falls ein bestimmter Seed/Config mal kein Layout erzeugt:
    for attempt in range(5):
        try:
            env.reset()
            print(f"[info] Env ready. Agents={env.get_num_agents()} Seed={args.seed} Try={attempt+1}")
            break
        except ValueError as e:
            if "no feasible environment" in str(e).lower():
                args.seed += 1
                env.rail_generator = sparse_rail_generator(
                    max_num_cities=max(2, args.cities),
                    grid_mode=False,
                    max_rails_between_cities=max(1, args.rails_between),
                    max_rail_pairs_in_city=max(1, args.rail_pairs_in_city),
                    seed=args.seed,
                )
                continue
            raise
    run_episode(env, args)


if __name__ == "__main__":
    main()
