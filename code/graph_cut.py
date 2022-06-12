import numpy as np
import cv2
import maxflow


class GraphGenerator(object):
    def __init__(self, img):
        # default constants
        self.foreground = 1
        self.background = 0
        self.sampled = 0
        self.segmented = 1
        self.default = 0.5
        self.maximum = 1e9

        # graph settings
        self.image = img
        self.segmented_image = None
        self.graph = None
        self.sample_cover = np.zeros_like(self.image)
        self.segment_cover = None
        self.mask = None
        self.background_samples = []
        self.foreground_samples = []
        self.nodes = []
        self.edges = []
        self.cur_cover = self.sampled

    def add_sample(self, x, y, mode):
        """Add sample points via GUI."""
        if mode == self.foreground:
            if (x, y) not in self.foreground_samples:
                self.foreground_samples.append((x, y))
                cv2.rectangle(
                    self.sample_cover,
                    (x - 1, y - 1),
                    (x + 1, y + 1),
                    (255, 16, 255),
                    -1,
                )
        elif mode == self.background:
            if (x, y) not in self.background_samples:
                self.background_samples.append((x, y))
                cv2.rectangle(
                    self.sample_cover,
                    (x - 1, y - 1),
                    (x + 1, y + 1),
                    (16, 255, 16),
                    -1,
                )

    def clear_samples(self):
        """Clear sample points."""
        self.foreground_samples = []
        self.background_samples = []
        self.sample_cover = np.zeros_like(self.image)

    def return_cover(self):
        """Return coverage map."""
        if self.cur_cover == self.sampled:
            return self.sample_cover
        elif self.cur_cover == self.segmented:
            return self.segment_cover

    def return_image_with_cover(self, mode):
        """Return image with coverage map."""
        if mode == self.sampled:
            return cv2.addWeighted(self.image, 0.8, self.sample_cover, 0.4, 0.1)
        elif mode == self.segmented:
            return cv2.addWeighted(self.image, 0.8, self.segment_cover, 0.4, 0.1)

    def gen_graph(self):
        """Create graph."""
        if len(self.foreground_samples) == 0 or len(self.background_samples) == 0:
            print('No sample points added.')
            return

        print("Generating graph...")
        self._seek_centers()
        self._fill_graph()
        self._graph_cut()
        print("Graph generated.")

    def _seek_centers(self):
        """Find centers of foreground and background samples."""
        height, width = self.image.shape[:2]
        self.graph = np.zeros((height, width))
        self.graph.fill(self.default)
        # self.foreground_center = np.mean(self.foreground_samples, axis=0)
        # self.background_center = np.mean(self.background_samples, axis=0)

        # sign of foreground samples
        for sample in self.foreground_samples:
            self.graph[sample[1] - 1, sample[0] - 1] = self.foreground

        # sign of background samples
        for sample in self.background_samples:
            self.graph[sample[1] - 1, sample[0] - 1] = self.background

    def _fill_graph(self):
        """Fill graph with nodes and edges."""
        self.nodes = []
        self.edges = []

        # connect all source vertexes to sink vertexes
        for (y, x), value in np.ndenumerate(self.graph):
            # for foreground samples
            if value == self.foreground:
                self.nodes.append((self.node_number(x, y), 0, self.maximum))

            # for background samples
            elif value == self.background:
                self.nodes.append((self.node_number(x, y), self.maximum, 0))

            else:
                self.nodes.append((self.node_number(x, y), 0, 0))

        for (y, x), value in np.ndenumerate(self.graph):
            if y == self.graph.shape[0] - 1 or x == self.graph.shape[1] - 1:
                continue
            idx = self.node_number(x, y)

            # neighbor node 1
            idx_1 = self.node_number(x + 1, y)
            g_1 = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y, x + 1], 2)))
            self.edges.append((idx, idx_1, g_1))

            # neighbor node 2
            idx_2 = self.node_number(x, y + 1)
            g_2 = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y + 1, x], 2)))
            self.edges.append((idx, idx_2, g_2))

    def _graph_cut(self):
        """Perform graph cut."""
        self.segment_cover = np.zeros_like(self.image)
        self.mask = np.zeros_like(self.image, dtype=bool)
        graph = maxflow.Graph[float](len(self.nodes), len(self.edges))
        nodelst = graph.add_nodes(len(self.nodes))

        # add tedges via nodes
        for node in self.nodes:
            graph.add_tedge(nodelst[node[0]], node[1], node[2])

        # add edges via edges
        for edge in self.edges:
            graph.add_edge(edge[0], edge[1], edge[2], edge[2])

        graph.maxflow()

        for idx in range(len(self.nodes)):
            if graph.get_segment(idx) == self.foreground:
                coord = self.node_coord(idx)
                self.segment_cover[coord[1], coord[0]] = (177, 154, 242)
                self.mask[coord[1], coord[0]] = (True, True, True)

    def swap_cover(self, mode):
        """Swap between sampled and segmented coverage map."""
        self.cur_cover = mode

    def gen_segmented_image(self):
        """Generate segmented image."""
        self.segmented_image = np.zeros_like(self.image)
        np.copyto(self.segmented_image, self.image, where=self.mask)

    def save_image(self):
        """Save segmented image."""
        if self.mask is None:
            print("Please segment the image first.")
            return

        out_filename = self.image_file.split(".")[0] + "_segmented.png"
        print(f'Saving to {out_filename}...')

        # save segmented image
        if self.segmented_image is not None:
            cv2.imwrite(out_filename, self.segmented_image)
        else:
            self.gen_segmented_image()
            cv2.imwrite(out_filename, self.segmented_image)

    def node_number(self, x, y):
        """Return node number."""
        return x + y * self.image.shape[1]

    def node_coord(self, node_num):
        """Return node coordinates."""
        return (node_num % self.image.shape[1], node_num // self.image.shape[1])
