import numpy as np

from wrappers.artist import random_relief_map, modify, figure_to_3drelief


def target_to_subtasks(figure):
    zh, xh, yh = figure.hole_indx
    xy_holes = np.asarray(list(zip(xh, yh)))
    targets_plane = figure.relief.astype(int)
    color_plane = None  # figure.figure_parametrs['color']
    X, Y = np.where(figure.relief != 0)
    addtional_tower_remote = (2,2)
    for x, y in zip(X, Y):
        for z in range(targets_plane[x, y]+3):
            custom_grid = np.zeros((9, 11, 11))
            if (color_plane is None) or (color_plane[z, x, y] == 0):
                custom_grid[z, x, y] = 1
                yield (x - 5, z -1, y - 5, 1), custom_grid
            else:
                custom_grid[z, x, y] = int(color_plane[z, x, y])
                yield (x - 5, z -1, y - 5, int(color_plane[z, x, y])), custom_grid

        if len(xy_holes) > 0 and x < (11 - addtional_tower_remote[0]) and y < (11 - addtional_tower_remote[1]):
            holes_in_xy = ((xy_holes - [x, y])[:, 0] == 0) & ((xy_holes - [x, y])[:, 1] == 0)
            holes_in_xy = np.where(holes_in_xy == 1)[0]
            additional_blocks = []
            last_height = 0
            z = 0
            for height in zh[holes_in_xy]:
               # raise Exception("!!!")
                print("HOLES in XY")
                print(zh[holes_in_xy])
                for z in range(last_height, height ):
                #    print("z, h", z, height - 1)
                    custom_grid = np.zeros((9, 11, 11))
                    custom_grid[z, x + addtional_tower_remote[0], y + addtional_tower_remote[1]] = 1
                    additional_blocks.append((z, x+addtional_tower_remote[0], y+addtional_tower_remote[1]))
                    yield (x - 5 + addtional_tower_remote[0], z - 1, y - 5 + addtional_tower_remote[1], 1), custom_grid
                custom_grid = np.zeros((9, 11, 11))

                custom_grid[height, x, y] = -1
                last_height = height
                yield (x - 5, height - 1, y - 5, -1), custom_grid
          #  print(additional_blocks)
            if len(additional_blocks)>0:
                for z,x,y in additional_blocks[::-1]:
                #    print("!! z, h", z, height - 1)
                    custom_grid = np.zeros((9, 11, 11))
                    custom_grid[z, x , y ] = -1
                    yield (x - 5, z - 1, y - 5, -1), custom_grid


def generate_preobs(min_value, max_value, red_degree = 5):
    choice_range = list(range(min_value,int(max_value)))
    p = 1/len(choice_range)
    p_for_bottom_block = p/red_degree
    addition_p = (p-p_for_bottom_block)/(len(choice_range)-1)
    p += addition_p
    probs = [p_for_bottom_block]+[p]*(len(choice_range)-1)
    return choice_range, probs

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
        # print("FIGURE")
        # print(figure_witn_colors.sum(axis = 0))
        # print()
        figure = np.zeros_like(figure_witn_colors)
        figure[figure_witn_colors > 0] = 1
        self.figure = figure.copy()
        holes, _ = modify(figure)
        _, _, full_figure = self.simplify()
        full_figure[full_figure!=0] = 1
        # print("FIGURE without holes")
        # print(full_figure.sum(axis = 0))
        # print()
        blocks = np.where((full_figure - self.figure )!=0)
        ind = np.lexsort((blocks[0], blocks[2], blocks[1]))
        self.hole_indx = (blocks[0][ind],blocks[1][ind],blocks[2][ind])
      #  print("DIFF")
     #   print((full_figure - self.figure ).sum(axis = 0))
    #    print()
        print("HOLES!")
     #   print()
        print(self.hole_indx)
        figure_parametrs = {'figure': self.figure, 'color': figure_witn_colors}
        self.figure_parametrs = figure_parametrs
        return figure

    def simplify(self):
        if self.figure is not None:
            fig = self.figure.copy()
            is_modified, new_figure = modify(fig)
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
        self.figures_height_range = (3, 8) if cnf is None else cnf['figures_height_range']
        self.std_range = (95, 160) if cnf is None else cnf['std_range']
        self.figures_count_range = (15, 30) if cnf is None else cnf['figures_count_range']
        self.color = color

    def make_task(self):
        plane = np.zeros((11, 11))
        choices, probs = generate_preobs(*self.figures_height_range)
        print(choices, probs)
        max_heigt = np.random.choice(choices, p = probs)
        relief = np.random.randint(1, max_heigt,
                                   size=(11, 11))
        x, y = np.random.randint(0, 11, size=2)
        relief_mask = random_relief_map(center=(x, y), std=np.random.randint(*self.std_range) / 100,
                                        count=np.random.randint(*self.figures_count_range))  # from artist
        plane[relief_mask] = 1
        relief[plane != 1] = 0
        fig_filter = np.mgrid[0:9, 0:11, 0:11][0] < (relief)
        figure = np.zeros((9, 11, 11))
        figure[fig_filter] = 1

        relief = figure.sum(axis = 0)
        high_blocks = np.where(relief > 1)
        holes_indx = [[],[],[]]

        for x, y in zip(*high_blocks):
            if relief[x,y] >= 3:
                if  relief[x,y] == 3:
                    count = np.random.randint(1,relief[x,y]-1)
                else:
                    count = np.random.randint(2, relief[x, y] - 1)
            else:
                count = 0
            orig_choice_range = list(range(0, int(relief[x, y])))
            choice_range = orig_choice_range.copy()
            for i in range(1, len(orig_choice_range)):
                choice_range += [orig_choice_range[i]] * (np.random.randint((i+1)*10, 100) )
            choice_range += [0]
            np.random.shuffle(choice_range)

            choice_range = choice_range[: count+3]
            print("original choice range", orig_choice_range)
            print("smth ",orig_choice_range[-len(orig_choice_range)//4:])
            print("cr", choice_range)
            choice_range = list(set(choice_range))
            holes_indx[0]  += sorted(choice_range)

            holes_indx[1] += [x]*len(choice_range)
            holes_indx[2] += [y]*len(choice_range)
            print("hi", holes_indx)
          # #  for i in range(count):
          #       choice_range = list(range(0,int(relief[x,y]-1)))
          #       np.random.shuffle(choice_range)
          #       p = 1/len(choice_range)
          #       p_for_bottom_block = p/4
          #       addition_p = (p-p_for_bottom_block)/(len(choice_range)-1)
          #       p += addition_p
          #       probs = [p_for_bottom_block]+[p]*(len(choice_range)-1)
          #       #print(probs)
          #
          #       z = np.random.choice(choice_range, p = probs)

        holes_indx = np.asarray(holes_indx)
        holes_indx[0]+=1

        if count == 0:
            holes_indx = [[],[],[]]
        print("count of holes: ", holes_indx[0])

    #    print(holes_indx)
   #     print(figure.shape)

        if len(holes_indx[0])>0:
            figure[holes_indx[0], holes_indx[1],holes_indx[2]] = 0
    #    print("GENERATED HOLES")
        holes_map = np.zeros((9,11,11))
        if len(holes_indx[0]) > 0:
            holes_map[holes_indx[0], holes_indx[1],holes_indx[2]] = 1
    #    print(holes_map.sum(axis = 0))
      #  print()
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
