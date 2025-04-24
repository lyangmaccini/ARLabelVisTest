import torch.optim as optim

from src.loss.loss import BCELossWithClassWeights
from src.metrics.helper import print_metrics
from src.metrics.metrics_calculator import MetricsCalculator
from src.wiring import get_source_data, get_training_data, get_model

import numpy as np
from data.binvox_rw import write, Voxels
import torch
from skimage import measure
import matplotlib.pyplot as plt
import pyvista as pv

def train_ours_neural(object_name, query, dimension, metrics_registry):
    print(f"oursNeural {object_name} {dimension}D {query} query")

    # hyperparameters
    n_regions = 50_000
    n_samples = 1500 if dimension == 4 else 500

    # load data
    data = get_source_data(object_name=object_name, dimension=dimension)

    # initialise model
    model = get_model(query=query, dimension=dimension)

    # initialise asymmetric binary cross-entropy loss function, and optimiser
    class_weight = 1
    criterion = BCELossWithClassWeights(positive_class_weight=1, negative_class_weight=1)
    optimiser = optim.Adam(model.parameters(), lr=0.0001)

    # initialise counter and print_frequency
    weight_schedule_frequency = 250_000
    # total_iterations = weight_schedule_frequency * 200  # set high iterations for early stopping to terminate training
    total_iterations = 30000
    evaluation_frequency = weight_schedule_frequency // 5
    print_frequency = 1000  # print loss every 1k iterations

    # instantiate count for early stopping
    count = 0

    for iteration in range(total_iterations):
        features, targets = get_training_data(data=data, query=query, dimension=dimension, n_regions=n_regions,
                                              n_samples=n_samples)

        # forward pass
        output = model(features)
        # print("feat")
        # print(features.shape)
        # print("target")
        # print(targets.shape)
        # print(output)

        # compute loss
        loss = criterion(output, targets)

        # zero gradients, backward pass, optimiser step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # print loss
        if (iteration + 1) % print_frequency == 0 or iteration == 0:
            print(f'Iteration: {iteration + 1}, Loss: {loss.item()}')

        if (iteration + 1) % evaluation_frequency == 0 or iteration == 0:
            prediction = (model(features).cpu().detach() >= 0.5).float().numpy()
            target = targets.cpu().detach().numpy()
            metrics = MetricsCalculator.calculate(prediction=prediction, target=target)
            print_metrics(metrics)

        if (iteration + 1) % evaluation_frequency == 0:
            prediction = (model(features).cpu().detach() >= 0.5).float().numpy()
            target = targets.cpu().detach().numpy()
            metrics = MetricsCalculator.calculate(prediction=prediction, target=target)

            # if convergence to FN 0 is not stable yet and still oscillating
            # let the model continue training
            # by resetting the count
            if count != 0 and metrics["false negatives"] != 0.:
                count = 0

            # ensure that convergence to FN 0 is stable at a sufficiently large class weight
            if metrics["false negatives"] == 0.:
                count += 1

            if count == 3:
                # save final training results
                metrics_registry.metrics_registry["oursNeural"] = {
                    "class weight": class_weight,
                    "iteration": iteration+1,
                    "false negatives": metrics["false negatives"],
                    "false positives": metrics["false positives"],
                    "true values": metrics["true values"],
                    "total samples": metrics["total samples"],
                    "loss": f"{loss:.5f}"
                }

                # early stopping
                print("early stopping\n")
                break

        # schedule increases class weight by 20 every 500k iterations
        if (iteration + 1) % weight_schedule_frequency == 0 or iteration == 0:
            if iteration == 0:
                pass
            elif (iteration + 1) == weight_schedule_frequency:
                class_weight = 20
            else:
                class_weight += 20

            criterion.negative_class_weight = 1.0 / class_weight

            print("class weight", class_weight)
            print("BCE loss negative class weight", criterion.negative_class_weight)
    # print(data.shape)

    print("Writing to file")
    final_features = []
    points = []
    dim = 32.0
    for x in range(int(dim)):
        for y in range(int(dim)):
            for z in range(int(dim)):
                points.append([x, y, z]) # points in the binvox space as a grid of 32x32x32 points
                final_features.append([x/dim, y/dim, z/dim]) # points scaled down to the [0, 1) space expected by the model
    
    final_pred = (model(torch.tensor(final_features)).cpu().detach() >= 0.5).float().numpy() # 0 or 1 predictions on each grid point
    final = np.zeros((int(dim), int(dim), int(dim)), dtype=np.bool8) # to store final voxels for binvox file
    for point, prediction in zip(points, final_pred):
        final[point[0]][point[1]][point[2]] = bool(prediction)
    
    verts, faces, normals, values = measure.marching_cubes(final, 0.0)
    # print("plotting")
    # print(verts)
    # print(verts.shape)
    # print(faces)
    # print(faces.shape)

    new_faces = []
    for face in faces:
        arr = np.insert(face, 0, 3)
        new_faces.append(arr)
    faces = np.hstack(new_faces)
    # print(faces)
    # print(faces.shape)

    mesh = pv.PolyData(verts, faces)
    mesh.subdivide(1, subfilter='linear', inplace=True)
    verts = mesh.points
    # print("new verts")
    # print(verts)
    # print(mesh.n_verts)
    # print(mesh.n_points)
    # print(verts.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(verts[:,0], verts[:,1], verts[:,2], c = "blue", alpha=0.5)
    ax.set_xlim([-2, 35])   # Set x-axis limits
    ax.set_ylim([-2, 35])   # Set y-axis limits
    ax.set_zlim([-2, 35])   # Set z-axis limits
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlabel("L")
    ax.set_ylabel("a")
    ax.set_zlabel("b")
    ax.set_title("Mesh Surface Plot")

    plt.show()

    filepath = "../../testNeuralBounding_" + str(int(dim)) + ".binvox"
    with open(filepath, 'w', encoding="latin-1") as fp:
        write(Voxels(final, [int(dim), int(dim), int(dim)], [0.0, 0.0, 0.0], 1.0, 'xyz'), fp) # write binvox file
