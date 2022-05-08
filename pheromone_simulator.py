from environment import Environment
from matplotlib import pyplot as plt
import cv2

env = Environment((32, 32), 5, 0.2, 0.1)
steps = 10

for i in range(steps):
    print("")
    print("-------------------------")
    print(f"Tick {i}")

    plt.imshow(env.get_visual_map())
    plt.show()
    #cv2.imwrite(f"states/{i}.jpg", env.get_visual_map())

    env.step()

