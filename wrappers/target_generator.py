import numpy as np

from wrappers.artist import random_relief_map, modify, figure_to_3drelief


def target_to_subtasks(figure):
    zh, xh, yh = figure.hole_indx
    xy_holes = np.asarray(list(zip(xh, yh)))
    targets_plane = figure.relief.astype(int)
    color_plane = None  # figure.figure_parametrs['color']
    X, Y = np.where(figure.relief != 0)
    for x, y in zip(X, Y):
        for z in range(targets_plane[x, y]):
            custom_grid = np.zeros((9, 11, 11))
            if (color_plane is None) or (color_plane[z, x, y] == 0):
                custom_grid[z, x, y] = 1
                yield (x - 5, z - 1, y - 5, 1), custom_grid
            else:
                custom_grid[z, x, y] = int(color_plane[z, x, y])
                yield (x - 5, z - 1, y - 5, int(color_plane[z, x, y])), custom_grid

        if len(xy_holes) > 0 and x < 10 and y < 10:
            holes_in_xy = ((xy_holes - [x, y])[:, 0] == 0) & ((xy_holes - [x, y])[:, 1] == 0)
            holes_in_xy = np.where(holes_in_xy == 1)[0]
            last_height = 0
            z = -1

            for height in zh[holes_in_xy]:
                print(zh[holes_in_xy])
                for z in range(last_height, height - 1):
                    print("z, h", z, height - 1)
                    custom_grid = np.zeros((9, 11, 11))
                    custom_grid[z, x + 1, y + 1] = 1
                    yield (x - 4, z - 1, y - 4, 2), custom_grid

                custom_grid = np.zeros((9, 11, 11))
                # if z!=0:
                z += 1
                custom_grid[z, x, y] = -1
                last_height = height
                yield (x - 5, z - 1, y - 5, -2), custom_grid


class Figure():
    def __init__(self, figure=None):
        self.use_color = True  # use color in figure generation
        self.figure = None  # figure without color
        self.figure_parametrs = None
        self.hole_indx = None  # all holes indexes
        self.simpl_holes = None  # holes only on the bottom
        self.relief = None  # 2d array of figure
        if figure:
            self.to_multitask_format(figure)

    def to_multitask_format(self, figure_witn_colors):
        figure = np.zeros_like(figure_witn_colors)
        figure[figure_witn_colors > 0] = 1
        self.figure = figure
        holes, _ = modify(figure)
        _, _, full_figure = self.simplify()
        self.hole_indx = np.where((figure == 0) & (full_figure != 0))
        figure_parametrs = {'figure': figure, 'color': figure_witn_colors}
        self.figure_parametrs = figure_parametrs
        return figure

    def simplify(self):
        if self.figure is not None:
            is_modified, new_figure = modify(self.figure)
            target, relief = figure_to_3drelief(new_figure)
            full_figure = relief.copy()
            relief = relief.max(axis=0)
            holes = relief - target.sum(axis=0)
            self.relief = relief
            self.simpl_holes = holes
        else:
            raise Exception("The figure is not initialized! Use 'make_task' method to do it!")
        return relief, holes, full_figure


class RandomFigure(Figure):
    def __init__(self, cnf=None, color=1):
        super().__init__()
        self.figures_height_range = (3, 9) if cnf is None else cnf['figures_height_range']
        self.std_range = (95, 160) if cnf is None else cnf['std_range']
        self.figures_count_range = (15, 30) if cnf is None else cnf['figures_count_range']
        self.color = color

    def make_task(self):
        plane = np.zeros((11, 11))
        relief = np.random.randint(1, np.random.randint(*self.figures_height_range),
                                   size=(11, 11))
        x, y = np.random.randint(0, 11, size=2)
        relief_mask = random_relief_map(center=(x, y), std=np.random.randint(*self.std_range) / 100,
                                        count=np.random.randint(*self.figures_count_range))  # from artist
        plane[relief_mask] = 1
        relief[plane != 1] = 0
        fig_filter = np.mgrid[0:9, 0:11, 0:11][0] < (relief)
        figure = np.zeros((9, 11, 11))
        figure[fig_filter] = 1

        blocks_index = np.where((figure != 0))
        count_of_blocks = blocks_index[0].shape[0]
        if count_of_blocks > 6:
            holes_count = np.random.randint(0, int(count_of_blocks * 0.7))
            holes_indx_filter = np.random.permutation(blocks_index[0].shape[0])[:holes_count]
            holes_indx = (blocks_index[0][holes_indx_filter],
                          blocks_index[1][holes_indx_filter],
                          blocks_index[2][holes_indx_filter])
            figure[holes_indx] = 0
        else:
            holes_indx = [[], [], []]
        self.hole_indx = holes_indx
        self.figure = figure
        self.simplify()
        self.figure_parametrs = {'figure': figure, 'color': figure * self.color, 'relief': self.relief}
        return figure


class DatasetFigure(Figure):
    def __init__(self, path_to_targets='../dialogue/augmented_targets.npy',
                 path_to_names='../dialogue/augmented_target_name.npy',
                 path_to_chats='../dialogue/augmented_chats.npy', generator=True, main_figure=None):
        super().__init__()
        from dialogue.model import TargetPredictor
        self.augmented_targets = np.load(path_to_targets)[:300]
        self.augmented_targets_names = np.load(path_to_names)[:300]
        self.augmented_chats = np.load(path_to_chats)[:300]
        self.target_predictor = TargetPredictor().cpu()
        if generator:
            self.generator = self.figures_generator()
        self.main_figure = main_figure

    def load_figure(self, idx, use_dialogue=True):
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
            rp = (fig2 - at).sum() == 0  # is figure right predicted
        else:
            figure_ = self.augmented_targets[idx]
            rp = 1  # is figure right predicted

        figure = self.to_multitask_format(figure_)
        self.figure_parametrs['name'] = name
        self.figure_parametrs['original'] = original
        self.figure_parametrs['right_predicted'] = rp
        self.figure_parametrs['relief'] = self.relief
        return figure

    def figures_generator(self, use_dialogue=True):
        i = 0
        while True:
            idx = i % len(self.augmented_targets)
            i += 1
            relief, holes, figure, name, rp, is_modified, original, f1_onstart = self.load_figure(idx, use_dialogue)

            yield relief, holes, figure, name, rp, is_modified, original, f1_onstart, self.augmented_chats[idx]
        return

    def make_task(self):
        if self.main_figure is not None:
            figure = self.load_figure(self.main_figure, True)
        else:
            figure = next(self.generator)
        return figure


if __name__ == "__main__":
    figure = RandomFigure()
    figure.make_task()
    generator = target_to_subtasks(figure)
    print(figure.relief)

    for i in range(25):
        task = next(generator)
        print()
        print("task is: ", task[0][-1])
        print(np.where(task[1] != 0))
