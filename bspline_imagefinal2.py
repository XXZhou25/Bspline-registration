import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from skimage.io import imread, imshow
from PIL import Image
from skimage.color import rgb2gray
from os.path import normpath as fn
# from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import resize
import math
from scipy.optimize import leastsq

from argparse import ArgumentParser

def bilinear_interpolation(x, y, points):           # Sorry I did not write this part but cited it from stackoverflow: https://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def spline(u, d):
    if (d == 0):
        return (-u**3 + 3*(u**2) - 3*u + 1) / 6
    elif d == 1:
        return (3*(u**3) - 6*(u**2) + 4) / 6
    elif d == 2:
        return (-3*(u**3) + 3*(u**2) +3*u +1)/6
    elif d == 3:
        return u**3 / 6

def bspline(i, j, gridspace) :

    u = i/gridspace
    v = j/gridspace
    # print([i, j])
    bdx = np.zeros([4, 1])
    bdx[0] = spline(u, 0)  # normalized distance
    bdx[1] = spline(u, 1)
    bdx[2] = spline(u, 2)
    bdx[3] = spline(u, 3)

    bdy = np.zeros([4, 1])
    bdy[0] = spline(v, 0)
    bdy[1] = spline(v, 1)
    bdy[2] = spline(v, 2)
    bdy[3] = spline(v, 3)

    one = np.ones([4, 1])
    bd1 = np.dot(bdy.T, one)
    bd = np.dot(bdx.T * bd1, one)

    return bdx, bdy, bd

def dx(newimage):           # gradient in x-axis direction
    dx1 = np.zeros(newimage.shape)
    dx1[:, 1:newimage.shape[1]] = newimage[:, 1:newimage.shape[1]] - newimage[:, 0:newimage.shape[1]-1]
    dx1[:, 0] = newimage[:, 0] - newimage[:, newimage.shape[1]-1]
    dx2 = np.zeros(newimage.shape)
    dx2[:, 0:newimage.shape[1]-1] = newimage[:, 1:newimage.shape[1]] - newimage[:, 0:newimage.shape[1]-1]
    dx2[:, newimage.shape[1]-1] = newimage[:, 1] - newimage[:, newimage.shape[1]-1]

    return (dx1 + dx2) / 2


if (__name__ == '__main__'):
    parser = ArgumentParser()
    parser.add_argument('-f', '--fixed', dest='fixed', help='fixed image name')
    parser.add_argument('-m', '--moving', dest='moving', help='moving image name')
    args = parser.parse_args()

    moving = imread(fn(args.moving))
    moving = resize(rgb2gray(moving), [300, 200])
    moving = 255 * moving
    # img = Image.fromarray(moving.astype(np.uint8), 'L')
    # img.save(fn('/Users/xiaoxiaozhou/Desktop/Research/ProfGeofferyHugo/BB.png'))

    fixed = imread(fn(args.fixed))
    fixed = resize(rgb2gray(fixed), [300, 200])
    fixed = 255 * fixed
    # img = Image.fromarray(fixed.astype(np.uint8), 'L')
    # img.save(fn('/Users/xiaoxiaozhou/Desktop/Research/ProfGeofferyHugo/AA.png'))

    d = 3
    gridspace = 10              #for 2-d image registration, sampling points are actually each pixel with interval 1
    iteration = 200
    lr = 0.001                  # learning rate
    lamda = 0.001               # regularization coefficient

    grids = np.zeros([(4+math.ceil(moving.shape[0]/gridspace)),(4+math.ceil(moving.shape[1]/gridspace)), 2])            # 34 * 34 control points, make sure every
    dpfield = np.random.random([(grids.shape[0])*gridspace, (grids.shape[1])*gridspace, 2])                             # 340 * 340 *2 desplacement field, save every pixel's x, y displacement
    newimage = np.zeros(moving.shape)
    bdmap = np.zeros(moving.shape)              # save every pixel's bspline
    plotloss = np.zeros(iteration)              # save every iteration's similarity
    bsplinex = np.zeros((10, 10, d+1))          # save x'direction bspline in one grid
    bspliney = np.zeros((10, 10, d+1))          # save y'direction bspline in one grid
    bsplined = np.zeros((10, 10))               # save tensor product of x,y spline in one grid

    # calculate and save each grid's spline
    for i in range(gridspace):
        for j in range(gridspace):
            bdx, bdy, bd = bspline(i, j, gridspace)
            bsplinex[i, j, :] = bdx[:,0]
            bspliney[i, j, :] = bdy[:,0]
            bsplined[i, j] = bd

    # registration
    for ite in range(iteration):

            for i in range(moving.shape[0]):
                for j in range(moving.shape[1]):

                    # x axis, closest control point (cpx1, cpy1) in left side of pixel(i, j)
                    cpx1 = (i + 2*gridspace) // (gridspace)
                    cpy1 = (j + 2*gridspace) // (gridspace)           # so 4 control points take right and down 4
                    tx = i - (i//gridspace)*gridspace                 # difference in each grid is normalized
                    ty = j - (j//gridspace)*gridspace
                    # find (i, j) spline
                    bdx = bsplinex[tx, ty, :]
                    bdy = bspliney[tx, ty, :]
                    bdmap[i, j] = bsplined[tx, ty]

                    cx = np.zeros([4, 4])
                    cy = np.zeros([4, 4])
                    for k1 in range(4):
                        for k2 in range(4):
                            cx[k1, k2] = dpfield[(cpx1-1+k1) * gridspace, (cpy1-1+k2) * gridspace, 0]           # when k1 = k2 = 0, the top left control point
                            cy[k1, k2] = dpfield[(cpx1-1+k1) * gridspace, (cpy1-1+k2) * gridspace, 1]           # this displacement is from original control points to new, instead of last time's to now's

                    #calculte (i, j)'s new displacement by spline method, tensor product
                    dpx1 = np.dot(bdx.T, cx)
                    dpx2 = np.dot(dpx1, bdy)

                    dpy1 = np.dot(bdx.T, cy)
                    dpy2 = np.dot(dpy1, bdy)

                    dpfield[i + 2 * gridspace, j + 2 * gridspace, 0] = dpx2
                    dpfield[i + 2 * gridspace, j + 2 * gridspace, 1] = dpy2

                    newi = i + dpy2
                    newj = j + dpx2

                    if newi < 0 or newi >= moving.shape[0] - 1 or newj < 0 or newj >= moving.shape[1] - 1:
                        if newi < 0:
                            newi = 0
                        elif newi >= moving.shape[0] - 1:
                            newi = moving.shape[0] - 1
                        if newj < 0:
                            newj = 0
                        elif newj >= moving.shape[1] - 1:
                            newj = moving.shape[1] - 1

                    # backward method find intensity in moving image
                    smalli = math.floor(newi)
                    smallj = math.floor(newj)
                    if smalli == moving.shape[0] - 1 or smallj == moving.shape[1] - 1:
                        newimage[i, j] = moving[smalli, smallj]
                    else:
                        surrounding = [(smalli, smallj, moving[smalli, smallj]),
                                       (smalli, smallj + 1, moving[smalli, smallj + 1]),
                                       (smalli + 1, smallj, moving[smalli + 1, smallj]),
                                       (smalli + 1, smallj + 1, moving[smalli + 1, smallj + 1])]
                        backints = bilinear_interpolation(newi, newj, surrounding)
                        newimage[i, j] = backints

            # newshow = newimage
            # toobig = np.where(newshow > 255)
            # newshow[toobig[0], toobig[1]] = 255
            # toosmall = np.where(newshow < 0)
            # newshow[toosmall[0], toosmall[1]] = 0
            # img = Image.fromarray(newshow.astype(np.uint8), 'L')
            # Image._show(img)

            similarity = np.sum(abs(newimage - fixed)) / (newimage.size * 255)

            dlossdi = 2 * (newimage - fixed)
            didx = dx(newimage)
            didy = (dx(newimage.T)).T
            du = np.sqrt(didx**2 + didy**2)

            didx2 = dx(didx)
            didy2 = (dx(didy.T)).T

            regularizor = -(didx2 + didy2)/(du+0.0001)

            dpfield[gridspace*2:newimage.shape[0]+gridspace*2,gridspace*2:newimage.shape[1]+gridspace*2, 0] = dpfield[gridspace*2:newimage.shape[0]+gridspace*2,gridspace*2:newimage.shape[1]+gridspace*2, 0] \
                                                                                                              - lr * dlossdi * didx * bdmap - lamda * regularizor * didx * bdmap

            dpfield[gridspace*2:newimage.shape[0]+gridspace*2,gridspace*2:newimage.shape[1]+gridspace*2, 1] = dpfield[gridspace*2:newimage.shape[0]+gridspace*2,gridspace*2:newimage.shape[1]+gridspace*2, 1] \
                                                                                                              - lr * dlossdi * didy * bdmap - lamda * regularizor * didy * bdmap

            plotloss[ite] = np.sqrt(similarity)
            print(similarity)


    img = Image.fromarray(newimage.astype(np.uint8), 'L')
    img.save(fn('/Users/xiaoxiaozhou/Desktop/Research/ProfGeofferyHugo/newnewnew.png'))
    plt.plot(np.arange(0, iteration), plotloss)
    plt.show()
