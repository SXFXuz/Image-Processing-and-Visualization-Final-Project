import numpy as np
import cv2
from graph_cut import GraphGenerator


class GCGUI(object):
    def __init__(self, img):
        # initialize the graph generator
        self.graph_gen = GraphGenerator(img)
        self.init_image = np.array(self.graph_gen.image)

        # initialize the GUI
        self.window = "Graph Cut Method"
        self.mode = None
        self.is_mouse_down = False

    def draw(self, event, x, y, *args):
        """Draw sample points."""
        # left mouse button down
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mode = self.graph_gen.foreground
            self.is_mouse_down = True
            self.graph_gen.add_sample(x - 1, y - 1, self.mode)

        # left mouse button up
        elif event == cv2.EVENT_LBUTTONUP:
            self.mode = None
            self.is_mouse_down = False

        # right mouse button down
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mode = self.graph_gen.background
            self.is_mouse_down = True
            self.graph_gen.add_sample(x - 1, y - 1, self.mode)

        # right mouse button up
        elif event == cv2.EVENT_RBUTTONUP:
            self.mode = None
            self.is_mouse_down = False

        # mouse move
        elif event == cv2.EVENT_MOUSEMOVE:
            # mouse button down
            if self.is_mouse_down:
                self.graph_gen.add_sample(x - 1, y - 1, self.mode)

    def run(self):
        """Run GUI."""
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.draw)

        while True:
            # show the image
            display = cv2.addWeighted(
                self.init_image, 0.8, self.graph_gen.return_cover(), 0.4, 0.1
            )
            cv2.imshow(self.window, display)

            # wait for a key press
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, quit
            if key == ord('q'):
                break

            # if the 'c' key is pressed, clear samples
            elif key == ord('c'):
                self.graph_gen.clear_samples()

            # if the 'm' key is pressed, show the mask image
            elif key == ord('m'):
                self.graph_gen.gen_graph()
                self.graph_gen.swap_cover(self.graph_gen.segmented)

            # if the 's' key is pressed, show the sampled image
            elif key == ord('s'):
                self.graph_gen.swap_cover(self.graph_gen.sampled)

        cv2.destroyAllWindows()

        self.graph_gen.gen_segmented_image()
        return self.graph_gen.segmented_image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="GCGUI", description="Graph Cut GUI", add_help=True
    )
    parser.add_argument('--input', '-i', type=str, help="input image", required=True)
    args = parser.parse_args()
    img = cv2.imread(args.input)
    gui = GCGUI(img)
    gui.run()
