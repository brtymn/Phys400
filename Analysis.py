import math
import numpy as np
import matplotlib.pyplot as plt

alpha_clip = 5

def Polar2Cartesian(r, theta):
    """theta in degrees

    returns tuple; (float, float); (x,y)
    """
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x,y


x1 = np.loadtxt('3input/x1.txt')
p1 = np.loadtxt('3input/p1.txt')
w1 = np.loadtxt('3input/w1.txt')

x2 = np.loadtxt('3input/x2.txt')
p2 = np.loadtxt('3input/p2.txt')
w2 = np.loadtxt('3input/w2.txt')

x3 = np.loadtxt('3input/x3.txt')
p3 = np.loadtxt('3input/p3.txt')
w3 = np.loadtxt('3input/w3.txt')

enc1 = np.loadtxt('3input/encoded_1.txt')
enc2 = np.loadtxt('3input/encoded_2.txt')
enc3 = np.loadtxt('3input/encoded_3.txt')

disp_1 = Polar2Cartesian(enc1[0] * alpha_clip, enc1[1])
print(disp_1)

disp_2 = Polar2Cartesian(enc2[0] * alpha_clip, enc2[1])
print(disp_2)

disp_3 = Polar2Cartesian(enc3[0] * alpha_clip, enc3[1])
print(disp_3)

fig = plt.figure(figsize=(12, 6), dpi=100)
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x1 + disp_1[0], p1 + disp_1[1], w1, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
ax.contour(x1 + disp_1[0], p1 + disp_1[1], w1, 10, cmap="RdYlGn", linestyles="solid", offset=-0.17)
ax.plot_surface(x2 + disp_2[0], p2 + disp_2[1], w2, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
ax.contour(x2 + disp_2[0], p2 + disp_2[1], w2, 10, cmap="RdYlGn", linestyles="solid", offset=-0.17)
ax.plot_surface(x3 + disp_3[0], p3 + disp_3[1], w3, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
ax.contour(x3 + disp_3[0], p3 + disp_3[1], w3, 10, cmap="RdYlGn", linestyles="solid", offset=-0.17)
ax.set_axis_off()
plt.show()
