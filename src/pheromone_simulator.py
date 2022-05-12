from environment import Environment
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from map import Map
import pickle
from PIL import Image
import os
from os import path

#map = Map(map_shape=(64, 64), n_food_sources=4, food_sigma=2, food_per_source=100, colony_size=5)

"""file = open('saved_maps/test_pickle', 'wb')
pickle.dump(map, file)"""

experiment_name = "experiment_0"
results_path = path.join("results", experiment_name)
map_path = path.join("saved_maps", experiment_name + ".pickle")


#diffusion_range = np.arange(0.49, 1.01, 0.01)
#diffusion_range = [0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
repetitions = 8
max_steps = 20000
for diffusion_value in diffusion_range:

    # make diffusion path if it doesn't already exist
    diffusion_path = path.join(results_path, str(diffusion_value))
    if not path.exists(diffusion_path):
        os.mkdir(diffusion_path)

    results_index_file = open(path.join(diffusion_path, "index.txt"), "a")

    for repetition in range(repetitions):

        # make repetition path if it doesn't already exist
        repetition_path = path.join(diffusion_path, str(repetition))
        if not path.exists(repetition_path):
            os.mkdir(repetition_path)

        with open(map_path, 'rb') as file:
            map = pickle.load(file)

        env = Environment(map, num_agents=100, diffusion_rate=diffusion_value, evaporation_rate=0.02)

        steps = 0
        while map.food_remaining and steps < max_steps:
            env.step()
            im = Image.fromarray(env.visualise())
            im.save(path.join(repetition_path, f"{steps}.png"))
            steps += 1

        results_index_file.write(f"{repetition} {steps}\n")
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

    results_index_file.close()


