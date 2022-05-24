import numpy as np
import matplotlib.pyplot as plt
import re


def drow_circle(img, R=5, coord=(10, 9)):
    width, height = img.shape[:2]
    x, y = coord
    X = np.arange(width).reshape(width, 1)
    Y = np.arange(height).reshape(1, height)
    mask = ((X - x) ** 2 + (Y - y) ** 2) < (R) ** 2
    return mask

def random_relief_map(center=3, std=1, count = 20):
    mask = np.random.normal(center, std, size = (count, 2)).astype(int)
    condition = np.where((mask[:, 0] <= 10) & (mask[:, 1] <= 10)
                         & (mask[:, 1] >= 0) & (mask[:, 0] >= 0))
    right_mask = mask[condition]
    return list(zip(*right_mask))

def map_from_img(img):
    img = plt.imread(img).mean(axis = 2)
    relief = np.random.randint(2, 3, size=(11, 11))
    relief[img==0]=0
    return relief

def figure_to_3drelief(target_):
   # print(np.where(target_!=0))
    target = np.zeros_like(target_)
    target[target_ != 0] = 1
    relief = target * np.arange(1,10).reshape(-1,1,1)
    return target, relief

def modif_tower(a):
 #   print("old tower -> ", a)
    idx = np.where(a==1)[0]
    diff =  idx[1:] - idx[:-1]
    holes = np.where(diff>1)
    modif = False
    if len(holes[0]) > 0:
        ones = np.where(a==1)[0][-1]
        a[idx[holes[0][0]]+1:ones] = 1
        modif = True
    return a,modif

def modify(figure):
    modifs = []
    new_figure = np.zeros_like(figure)
    modifs = False
    for i in range(figure.shape[1]):
        for j in range(figure.shape[2]):
            tower = figure[:,i,j]
            tower[tower>0]=1
            new_figure[:,i,j], flag = modif_tower(tower)
            modifs|= flag
            binary = "".join(str(figure[:,i,j]).split(" "))
            binary = re.sub('[123456789]', '1', binary)
            p = re.findall("10+1", binary)
            if len(p) > 0:
                modifs.append([i,j])
    return modifs, new_figure



if __name__=="__main__":
    pass
 #    relief, holes = iglu()
 #
 #  #  plt.imshow(relief)
 #  #  plt.show()
 # #   holes = do_hole_map(relief)
 #    #generator = relief_to_target(relief)
 #
 #    #next(generator)
 #
 #
 #          #  generator = relief_to_target(relief)
 #    plt.subplot(2,1,1)
 #    plt.imshow(relief)
 #    plt.colorbar()
 #    plt.subplot(2, 1, 2)
 #    plt.imshow(holes)
 #    plt.colorbar()
 #
 #   # plt.imshow(relief)
 #    plt.show()
