from environment import Environment
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from map import Map

map = Map(map_shape=(64, 64), n_food_sources=4, food_sigma=2, food_per_source=100, colony_size=5)

env = Environment(map, num_agents=100, diffusion_rate=0.3, evaporation_rate=0.02)
steps = 100

plt.imshow(env.visualise())
plt.show()

frames = [env.visualise()]
for i in range(steps):
    if i % 100 == 0:
        print(f"Step: {i}")

    env.step()
    frames.append(env.visualise())
    # print(cv2.imwrite(f"states/{i}.jpg", env.get_visual_map()))

frames = np.array(frames).astype(np.uint8)


# Save as mp4
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
ani.save('movie.mp4')
plt.show()


