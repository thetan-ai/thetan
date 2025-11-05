import numpy as np
from fury import window, actor, colormap, material

# Generate smooth terrain
def gaussian(x, y, x0, y0, sigma, height):
    return height * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def sample_gaussian(x0, y0, sigma, height):

    return np.random.normal
    return height * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)
Z = (gaussian(X, Y, -3, -3, 2, 3) +
     gaussian(X, Y, 4, 4, 3, 2.5) +
     gaussian(X, Y, 0, 0, 1.5, 6) -
     gaussian(X, Y, -5, 5, 3, 1.5))

x = np.linspace(-15, 15, 300)
y = np.linspace(-15, 15, 300)
X, Y = np.meshgrid(x, y)
Z = (gaussian(X, Y, -6, -6, 2, 3) +
     gaussian(X, Y, 10, 10, 3, 8) +
     gaussian(X, Y, 0, 0, 1.5, 6) -
     gaussian(X, Y, -5, 5, 3, 9))



# Construct vertices and triangle faces
vertices = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))



faces = []
n = X.shape[0]
for i in range(n - 1):
    for j in range(n - 1):
        idx = i * n + j
        faces.append([idx, idx + 1, idx + n])
        faces.append([idx + 1, idx + n + 1, idx + n])
faces = np.array(faces)

# Optional: replace with your own TB centroids
optima = np.array([
    [0, 0, 4],
    [-3, -3, 3],
    [4, 4, 2.5],
    [-5, 5, -1.5],
])

theta = 3

points = vertices.copy()


points[:, 2] = 5.05 * points[:, 2]

from thetan.berserker import berserker2_clean
from thetan.theta import theta_g1

ind = np.random.choice(np.arange(points.shape[0]), points.shape[0]//100, replace=False)
points = points[ind].copy()

centroids, labels = berserker2_clean(points, theta_value=theta)
centroids2, labels2 = theta_g1(points, theta=theta)


# Map heights to terrain colormap
colors = colormap.create_colormap(vertices[:, 2], name='plasma')
surf_actor = actor.surface(vertices, faces, colors=colors)
# material.manifest_pbr(surf_actor)
surf_actor.GetProperty().SetOpacity(1.)


# Red spheres for centroids
optima_actors = actor.sphere(centers=optima, colors=(1, 0, 0), radii=0.3)
centroid_actors = actor.sphere(centers=centroids, colors=(0.5, 0.5, 1), radii=0.1)
centroid_actors2 = actor.sphere(centers=centroids2, colors=(0.9, 0.5, 0.1), radii=0.1)



# Create the scene
scene = window.Scene()
scene.background((0.1, 0.1, 0.1))
scene.background((1, 1, 1))

scene.add(surf_actor)

# scene.add(optima_actors)
scene.add(centroid_actors)
# scene.add(centroid_actors2)

scene.set_camera(position=(0, -40, 30), focal_point=(0, 0, 0), view_up=(0, 0, 1))


# Show interactive window
window.show(scene, size=(800, 800), reset_camera=False)
