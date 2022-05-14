from environment import Environment
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from map import Map
import pickle
from PIL import Image
import os
from os import path
import multiprocessing as mp


def run_experiment(experiment_name, diffusion_value, max_steps):
    """
    Run an experiment on the simulation
    :param experiment_name: The name of the experiment
    :param diffusion_value: The value of the diffusion. Should be between 0 and 1 for sensible results
    :param max_steps: The maximum number of steps of the experiment before halting.
    :return: None
    """
    if not path.exists("results"):
        os.mkdir("results")

    results_path = path.join("results", experiment_name)

    if not path.exists(results_path):
        os.mkdir(results_path)

    map_path = path.join("saved_maps", experiment_name + ".pickle")

    # make diffusion path if it doesn't already exist
    diffusion_path = path.join(results_path, "{:.4f}".format(round(diffusion_value, 2)))
    if not path.exists(diffusion_path):
        os.mkdir(diffusion_path)

    if path.exists(map_path):
        with open(map_path, 'rb') as file:
            map = pickle.load(file)
    else:
        map = Map(map_shape=(128, 128), n_food_sources=3, food_sigma=2, food_per_source=1000, colony_size=5)
    env = Environment(map, num_agents=1200, diffusion_rate=diffusion_value, evaporation_rate=0.02)
    steps = 0
    frames = []
    while map.food_remaining and steps < max_steps:
        if steps % 100 == 0:
            print(f"Step: {steps}")
        env.step()
        frames.append(env.visualise())
        steps += 1
    with open(path.join(diffusion_path, "index.txt"), "a+") as file:
        file.write(f"{steps}\n")

    # Save as mp4
    frames = np.array(frames).astype(np.uint8)
    mat_frames = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis("off")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    for frame in frames:
        mat_frames.append([plt.imshow(frame, animated=True)])
    ani = animation.ArtistAnimation(fig, mat_frames, interval=50, blit=True, repeat_delay=1000)
    ani.save(f'{diffusion_value}.mp4')


if __name__ == "__main__":
    diffusion_range = [0.26347]

    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)

    for repetition in range(1):
        for diffusion_rate in diffusion_range:
            pool.apply_async(run_experiment, args=("experiment_1", diffusion_rate, 100))
    pool.close()
    pool.join()
