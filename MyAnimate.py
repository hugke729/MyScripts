import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import ImageSequenceClip
import numpy as np


def animate(loop_func, loc='output', fps=1, imin=0, imax=np.inf, istep=1):
    """Use moviepy to create an animation from a for loop

    Inputs
    ------
    loop_func: a function that takes only the input i and returns a figure
    loc: filename to save within /home/hugke729/
    fps: frames per second for output movie
    imin: starting index
    imax: last index
    istep: create frame every ith time"""

    plt.ioff()
    frame_list = []

    i = imin
    frames_made = 0  # Count number of frames made to update user on progress
    print('Frames completed:', end=' ', flush=True)

    while True:
        # Plot
        try:
            fig = loop_func(i)

            # Confirm correct plot
            # NEEDS WORK
            # if i == imin:
            #     plt.imshow(rgb_fig)
            #     plt.show()
            #     check = input('Press z to cancel')
            #     if check.lower() == z:
            #         break

            # Update i for next loop
            i += istep

        except IndexError:
            print('\nNo more frames to create')
            plt.close()
            break

        # Convert to image and add to list
        rgb_fig = mplfig_to_npimage(fig)
        frame_list.append(rgb_fig)

        if i > imax:
            print('\nStopping at imax')
            break

        # Update progress to user
        frames_made += 1
        if frames_made % 10 == 0:
            print(frames_made, flush=True, end=' ')
        plt.close()

    clip = ImageSequenceClip(frame_list, durations=[1.0/fps]*frames_made)
    clip.write_videofile('/home/hugke729/' + loc + '.mp4', fps=fps)

    # Return to interactive plotting
    plt.ion()
