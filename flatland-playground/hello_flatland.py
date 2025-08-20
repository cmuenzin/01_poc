from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.utils.rendertools import RenderTool


# Robustere Parameter: min. 2 Städte + etwas größerer Grid
rail_gen = sparse_rail_generator(
    max_num_cities=2,             # mindestens 2 Städte (Start/Ziel)
    grid_mode=False,
    max_rails_between_cities=2,
    max_rail_pairs_in_city=2,
    seed=7
)

env = RailEnv(
    width=25, height=25,
    number_of_agents=1,
    rail_generator=rail_gen,
    line_generator=sparse_line_generator(),  # (früher: schedule_generator)
    obs_builder_object=GlobalObsForRailEnv()
)

# Retry-Loop: falls ein Seed mal "kein Layout möglich" ergibt
for attempt in range(10):
    try:
        obs, info = env.reset()
        print("Flatland OK. Agents:", env.get_num_agents(), "Try:", attempt+1)
        break
    except ValueError as e:
        if "no feasible environment" in str(e).lower():
            # Seed ändern und erneut versuchen
            rail_gen = sparse_rail_generator(
                max_num_cities=2,
                grid_mode=False,
                max_rails_between_cities=2,
                max_rail_pairs_in_city=2,
                seed=7 + attempt + 1
            )
            env.rail_generator = rail_gen
        else:
            raise

from flatland.utils.rendertools import RenderTool
import numpy as np, time
# Render-Tool initialisieren
renderer = RenderTool(env, gl="PGL", screen_width=800, screen_height=800)

for t in range(60):
    actions = {a: np.random.randint(0, 5) for a in range(env.get_num_agents())}
    obs, rewards, done, info = env.step(actions)
    renderer.render_env(show=True, show_observations=False, show_predictions=False)
    if done["__all__"]:
        break
    time.sleep(0.03)
print("Render fertig.")
