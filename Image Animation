DRIVING_VIDEO = "dameyo.mp4"
GAMBAR_TARGET = "imamjoming.jpg"
OUTPUT = "imamjoming_dameyo.mp4"

!git clone https://github.com/AliaksandrSiarohin/first-order-model

cd /content/first-order-model

from google.colab import drive
drive.mount('/content/gdrive')

import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

source_image = imageio.imread('/content/gdrive/My Drive/first-order-motion-model/{}'.format(GAMBAR_TARGET))
driving_video = imageio.mimread('/content/gdrive/My Drive/first-order-motion-model/{}'.format(DRIVING_VIDEO), memtest=False)


#Resize image and video to 256x256

source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani
    

HTML(display(source_image, driving_video).to_html5_video())

"""**Create a model and load checkpoints**"""

from demo import load_checkpoints
generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', 
                            checkpoint_path='/content/gdrive/My Drive/first-order-motion-model/vox-cpk.pth.tar')

"""**Perform image animation**"""

from demo import make_animation
from skimage import img_as_ubyte

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

#save resulting video
imageio.mimsave('/content/gdrive/My Drive/first-order-motion-model/{}'.format(OUTPUT), [img_as_ubyte(frame) for frame in predictions])
#video can be downloaded from drive folder

HTML(display(source_image, driving_video, predictions).to_html5_video())
