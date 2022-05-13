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
    results_path = path.join("results", experiment_name)
    map_path = path.join("saved_maps", experiment_name + ".pickle")

    # make diffusion path if it doesn't already exist
    diffusion_path = path.join(results_path, "{:.2f}".format(round(diffusion_value, 2)))

    if not path.exists(diffusion_path):
        os.mkdir(diffusion_path)

    with open(map_path, 'rb') as file:
        map = pickle.load(file)

    env = Environment(map, num_agents=1200, diffusion_rate=diffusion_value, evaporation_rate=0.02)
    steps = 0
    while map.food_remaining and steps < max_steps:
        env.step()
        #im = Image.fromarray(env.visualise())
        #im.save(path.join(repetition_path, f"{steps}.png"))
        steps += 1
    with open(path.join(diffusion_path, "index.txt"), "a+") as file:
        file.write(f"{steps}\n")
    #frames = np.array(frames).astype(np.uint8)

    # Save as mp4
    """mat_frames = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis("off")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    for frame in frames:
        mat_frames.append([plt.imshow(frame, animated=True)])

    ani = animation.ArtistAnimation(fig, mat_frames, interval=50, blit=True, repeat_delay=1000)
    ani.save('movie.mp4')"""
    # plt.show()


if __name__ == "__main__":

    diffusion_range = np.arange(0, 1.01, 0.01)

    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)

    for repetition in range(100):
        for diffusion_rate in diffusion_range:
            pool.apply_async(run_experiment, args=("experiment_1", diffusion_rate, 20000))
    pool.close()
    pool.join()

    # map = Map(map_shape=(64, 64), n_food_sources=4, food_sigma=2, food_per_source=100, colony_size=5)

    """file = open('saved_maps/test_pickle', 'wb')
    pickle.dump(map, file)"""


    #
    # diffusion_range = [0.95, 0.96, 0.97, 0.98, 0.99, 1.00]

    """map = Map(map_shape=(128, 128), n_food_sources=3, food_sigma=2, food_per_source=1000, colony_size=5)
    plt.imshow(map.visualise())
    plt.show()"""

    """file = open('saved_maps/experiment_1.pickle', 'wb')
    pickle.dump(map, file)"""



