import numpy as np
from wrappers.artist import random_relief_map, drow_circle, modify, figure_to_3drelief
from dialogue.model import TargetPredictor


def target_to_subtasks(targets_plane, hole_plane=None, color_plane=None):
    X, Y = np.where(targets_plane >= 1)
    for x, y in zip(X, Y):
        for z in range(targets_plane[x, y]):
            custom_grid = np.zeros((9, 11, 11))
            if (color_plane is None) or (color_plane[z, x, y] == 0):
                custom_grid[z, x, y] = 1
                yield (x - 5, z - 1, y - 5, 1), custom_grid
            else:
                custom_grid[z, x, y] = int(color_plane[z, x, y])
                yield (x - 5, z - 1, y - 5, int(color_plane[z, x, y])), custom_grid
        if hole_plane is not None:
            try:
                houl_depth = hole_plane[x, y]
            except:
                raise Exception("Something wrong!")
            for z in range(houl_depth):
                custom_grid = np.zeros((9, 11, 11))
                custom_grid[z, x, y] = -1
                yield (x - 5, z - 1, y - 5, -1), custom_grid

class RandomFigure():
    def __init__(self, cnf = None):
        self.figures_height_range = (3, 8) if cnf is None else cnf['figures_height_range']
        self.std_range = (95, 160) if cnf is None else cnf['std_range']
        self.figures_count_range = (15, 30) if cnf is None else cnf['figures_count_range']
        self.relief, self.hole_relief = self.make_task()
        self.use_color = False
        #return self.relief, self.hole_relief

    def make_task(self):
        ###make relief
        plane = np.zeros((11, 11))
        relief = np.random.randint(1, np.random.randint(*self.figures_height_range),
                                   size=(11, 11))
        x, y = np.random.randint(0, 12, size=2)
        relief_mask = random_relief_map(center=(x, y), std=np.random.randint(*self.std_range) / 100,
                                        count=np.random.randint(*self.figures_count_range)) #from artist
        plane[relief_mask] = 1
        relief[plane != 1] = 0

        field_center = (5, 5)
        center_mask = drow_circle(relief, R=2, coord=field_center) #from artist
        relief[center_mask] = 0

        ###make holes
        hole_relief = np.zeros_like(relief)
        X, Y = np.where(relief > 1)
        idexes = list(range(len(X)))
        for i in idexes:
            hole_relief[X[i], Y[i]] = np.random.randint(0, relief[X[i], Y[i]])
        self.relief = relief
        self.hole_relief = hole_relief

        return relief, hole_relief


class DatasetFigure():

    def __init__(self, path_to_targets = '../dialogue/augmented_targets.npy',
                 path_to_names = '../dialogue/augmented_target_name.npy',
                 path_to_chats = '../dialogue/augmented_chats.npy', generator = True, main_figure = None):

        self.augmented_targets = np.load(path_to_targets)[:300]
        self.augmented_targets_names = np.load(path_to_names)[:300]
        self.augmented_chats = np.load(path_to_chats)[:300]
        self.use_color = True
        self.target_predictor = TargetPredictor().cpu()
        if generator:
            self.generator = self.figures_generator()
        self.main_figure = main_figure
        self.relief, self.hole_relief = self.make_task()

    def load_figure(self, idx, use_dialogue = True):
        name = self.augmented_targets_names[idx]
        at = self.augmented_targets[idx][:, :, :]
        at[self.augmented_targets[idx] > 0] = 1
        original = self.augmented_targets[idx]
        original_ones = np.zeros_like(original)
        original_ones[original > 0] = 1
        if use_dialogue:
            figure_ = self.target_predictor(self.augmented_chats[idx]).detach().numpy()
            fig2 = figure_[:, :, :]
            fig2[fig2 > 0] = 1
            rp = (fig2 - at).sum() == 0
        else:
            figure_ = self.augmented_targets[idx]
            rp = 1
        figure = np.zeros_like(figure_)
        figure[figure_ > 0] = 1
        is_modified, new_figure = modify(fig2)
        target, relief = figure_to_3drelief(new_figure)

        maximal_intersection = (original_ones * new_figure).sum()
        current_grid_size = new_figure.sum()
        target_grid_size = original_ones.sum()
        curr_prec = maximal_intersection / target_grid_size
        curr_rec = maximal_intersection / current_grid_size
        if maximal_intersection == 0:
            f1_onstart = 0
        else:
            f1_onstart = 2 * curr_prec * curr_rec / (curr_rec + curr_prec)

        relief = relief.max(axis=0)
        holes = relief - target.sum(axis=0)
        return relief, holes, figure, name, rp, is_modified, original, f1_onstart

    def figures_generator(self, id=None, use_dialogue=True):
        i = 0
        while True:
            idx = i % len(self.augmented_targets)
            i += 1
            relief, holes, figure, name, rp, is_modified, original, f1_onstart = self.load_figure(idx, use_dialogue)

            yield relief, holes, figure, name, rp, is_modified, original, f1_onstart, self.augmented_chats[idx]
        return 0

    def make_task(self):
        if self.main_figure is not None:
            figure = self.load_figure(self.main_figure, True)
        else:
            figure = next(self.generator)
        relief = figure[0]
        holes = figure[1]
        self.color = figure[2]
        self.name = figure[3]
        self.is_right_predicted = figure[4]
        self.is_modified = figure[5]
        self.original = figure[6]
        self.f1_onstart = figure[7]
        self.chat = figure[8]

        return relief, holes

