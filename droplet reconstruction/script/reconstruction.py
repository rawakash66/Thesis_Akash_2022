# importing libraries
import os
import vtk
import cv2
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from scipy.io import savemat, loadmat
from plyfile import PlyData, PlyElement

sm.set_framework('tf.keras')

# function to get 3D points from circle pattern
def get_points_circle(img_path, n_circles, c_dist, width, height):
    '''
    Parameters
    ----------
    img_path : string
        path to folder containing pattern images
    n_circles : int
        total number of circles in circle pattern divisible by 11.
    c_dist : int
        distance between two circles divisible by 2
    width : int
        number of corner in horizontal direction.
    height : int
        number of corners in vertical direction.

    Returns
    -------
    3D object points.
    '''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # setup SimpleBlobDetector parameters.
    blobParams = cv2.SimpleBlobDetector_Params()
    
    # change thresholds
    blobParams.minThreshold = 8
    blobParams.maxThreshold = 255
    
    # filter by area
    blobParams.filterByArea = True
    blobParams.minArea = 64
    blobParams.maxArea = 2500
    
    # filter by circularity
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.1
    
    # filter by convexity
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.87
    
    #filter by inertia
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01
    
    # create detector with the above params
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)
    
    # create 3D points assuming all blobs are in XY plane and distance between
    # two circles is c_dist units (can be replaced by any number, real size is pointless)
    objp = np.zeros((n_circles, 3), np.float32)
    itr = 0
    col = 0
    while itr < n_circles:
        if (col % 2 == 0):
            for row in range(n_circles//11):
                objp[itr + row] = (c_dist*(col//2), c_dist*row, 0)
        else:
            for row in range(n_circles//11):
                objp[itr + row] = (c_dist//2 + c_dist*((col-1)//2), c_dist//2 + c_dist*row, 0)
        col += 1
        itr += (n_circles//11)
    
    # arrays to store 3D points and 2D points from all images
    objpoints = [] # 3D
    imgpoints = [] # 2D
    
    images = sorted(os.listdir(img_path))
    
    # iterate through all images
    found = 0
    
    for fname in images:
        img = cv2.imread(img_path + fname, cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # detect blobs
        keypoints = blobDetector.detect(gray)
        
        # draw detected blob as red circle to help cv2.findCirclesGrid() function
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
        
        # find circles
        ret, corners = cv2.findCirclesGrid(im_with_keypoints, (width, height), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)
        
        # if corners found add 3D and 2D points
        if ret:
            found += 1
            objpoints.append(objp)
            corners2D = cv2.cornerSubPix(im_with_keypoints_gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2D)
            
            # draw keypoints
            im_with_keypoints = cv2.drawChessboardCorners(img, (width, height), corners2D, ret)
            cv2.imwrite("../data/pose/tracked_points/" + fname, im_with_keypoints)
    
    print("Number of images found: ", found)
    
    return objpoints, imgpoints

# function to load intrinsic parameters
def load_intrinsics(path):
    '''
    Parameters
    ----------
    path : string
        location of stored data.

    Returns
    -------
    camera intrinsic parameters.
    '''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    intrinsic_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    
    return intrinsic_matrix, dist_matrix

# function to get camera projection and camera pose from 3D-2D correspondences and intrinsic matrices
def save_projections_and_pose(objpoints, imgpoints, intrinsic, distCoeff, newmtx, path):
    '''
    Parameters
    ----------
    objpoints : list
        list of 3D points for each image.
    imgpoints : list
        list of 2D correspondence points for each image.
    intrinsic : np.array
        3 x 3 camera intrinsic matrix.
    distCoeff : np.array
        lens distortion coefficients.
    newmtx : np.array
        3 X 3 optimal intrinsic matrix
    path : str
        location to save projections and poses

    Returns
    -------
    camera projections and position.
    '''
    # list for projections and poses
    poses = []
    projections = []
    
    # solve PnP problem and get pose and projection for each image
    for objp, imgp in zip(objpoints, imgpoints):
        _, rvec, tvec = cv2.solvePnP(objp, imgp, intrinsic, distCoeff)
        R = cv2.Rodrigues(rvec)[0]
        c = np.dot(-R.T, tvec)
        P = np.dot(newmtx, np.hstack((R, tvec)))
        projections.append(P)
        poses.append(c)
        
    # dictionary to save
    mat_dict = {'P': projections, 'c': poses}
    savemat(path, mat_dict)

# function to load camera projections and images
def load_data(proj_path, img_path):
    '''
    Parameters
    ----------
    proj_path : string
        path to the projection data folder.
    img_path : string
        path to the image folder

    Returns
    -------
    camera projections and images.
    '''
    # load camera projection matrices
    data = loadmat(proj_path)
    proj = data['P']
    projections = [proj[i, :, :] for i in range(proj.shape[0])]
    
    # load images
    files = sorted(os.listdir(img_path))
    images = []
    
    for f in files:
        img = cv2.imread(img_path + f, cv2.IMREAD_UNCHANGED).astype(np.float32)
        images.append(img)
    
    return projections, images

# function to undistort images
def undistort_images(dist_images, mtx, newmtx, dist):
    '''
    Parameters
    ----------
    dist_images : list
        list of distorted images.
    mtx : np.array
        3 X 3 intrinsic matrix.
    newmtx : np.array
        optimal intrinsic matrix.
    dist : np.array
        distortion coefficients.

    Returns
    -------
    new camera matrix and undistorted images.
    '''
    H, W, _ = dist_images[0].shape
    images = []
    
    for img in dist_images:
        dst = cv2.undistort(img, mtx, dist, None, newmtx)
        images.append(dst)
    
    return images

# function to get silhouette from images
def get_silhouette(images, path):
    '''
    Parameters
    ----------
    images : list
        list of images containing the captured images.
    path : string
        path to the deep learning model file for predicting the silhouette mask.

    Returns
    -------
    silhoutte images.
    '''
    H, W, _ = images[0].shape
    silhouette = []
    
    # load U-Net model
    model = sm.Unet('efficientnetb4', input_shape = (224, 224, 3), classes = 1, activation = 'sigmoid')
    model.load_weights(path).expect_partial()
    
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgx = cv2.resize(img, (224, 224), interpolation = cv2.INTER_NEAREST_EXACT)
        img_input = np.expand_dims(imgx, axis = 0)
        pred = model.predict(img_input).round()
        pred_mask = pred[..., 0].squeeze()
        final_mask = cv2.resize(pred_mask, (W, H), interpolation = cv2.INTER_NEAREST_EXACT)
        final_mask = final_mask.round()
        silhouette.append(final_mask)
    
    return silhouette

# function to create voxel grid of points
def create_voxel(size, scale_factor):
    '''
    Parameters
    ----------
    size : int
        size of the voxel cube.
    scale_factor : tuple
        three int values to scale each axis points

    Returns
    -------
    3D voxel points.
    '''
    # create meshgrid of points
    x, y, z = np.mgrid[:size, :size, :size]
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(float)
    points = points.T
    
    # get number of points
    n_points = points.shape[0]
    
    # normalize points with max values
    xmax, ymax, zmax = np.max(points, axis = 0)
    points[:, 0] /= xmax
    points[:, 1] /= ymax
    points[:, 2] /= zmax
    
    # get center of the voxel grid and centralise the voxels
    center = points.mean(axis = 0)
    points -= center
    
    # get min
    xmin, ymin, zmin = np.min(points, axis = 0)
    
    # translate points
    points[:, 2] -= zmin
    points[:, 1] -= ymin
    points[:, 0] += xmin
    
    # scale the points
    points[:, 0] *= scale_factor[0]
    points[:, 1] *= scale_factor[1]
    points[:, 2] *= scale_factor[2]
    # points[:, 2] -= 10
    
    # convert to homogenous coordinates
    points = np.vstack((points.T, np.ones((1, n_points))))
    
    return points

# function to carve the voxels
def start_carving(projections, silhouettes, points):
    '''
    Parameters
    ----------
    projections : list
        list of 3 x 4 camera projection matrices.
    silhouettes : list
        list of silhouette images.
    points : np.array
        array of homogenous 3D points representing points of voxel grid.

    Returns
    -------
    occupancy matrix.
    '''
    # filled list to store value of image points for each image
    H, W = silhouettes[0].shape
    filled = []
    
    # iterate over each image
    for P, img in zip(projections, silhouettes):
        
        # 2D image points
        uvs = P @ points
        uvs /= uvs[2, :]
        uvs = np.round(uvs).astype(int)
        
        # get good points
        x_good = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < W)
        y_good = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < H)
        good = np.logical_and(x_good, y_good)
        
        # get value of the points inside the image
        indices = np.where(good)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = uvs[:2, indices]
        res = img[sub_uvs[1, :], sub_uvs[0, :]]
        fill[indices] = res
        
        # add to the filled list
        filled.append(fill)
    
    # get numpy array
    filled = np.vstack(filled)
    
    # scalar occupancy matrix storing points with their occupancy values
    occupancy = np.sum(filled, axis = 0)
    
    return occupancy

# function to form a rectilinear grid with occupancy scalars
def get_rectilinear_grid(size, points, occupancy):
    '''
    Parameters
    ----------
    size : int
        size of cubic voxel grid.
    points : np.array
        array of dimensions 4 x n_points representing homogenous 3D coordinates of grid.
    occupancy : np.array
        matrix containing occupancy value of each voxel.

    Returns
    -------
    vtk rectilinear grid object.
    '''
    # generating x, y and z points
    pts = points.T
    x = pts[::size*size, 0]
    y = pts[:size*size:size, 1]
    z = pts[:size, 2]
    
    # generate x grid using vtk library
    xCoords = vtk.vtkFloatArray()
    for i in x:
        xCoords.InsertNextValue(i)
    
    # generate y grid using vtk library
    yCoords = vtk.vtkFloatArray()
    for i in y:
        yCoords.InsertNextValue(i)
    
    # generate z grid using vtk library
    zCoords = vtk.vtkFloatArray()
    for i in z:
        zCoords.InsertNextValue(i)
    
    # generate scalar occupancy value using vtk library
    values = vtk.vtkFloatArray()
    for i in occupancy:
        values.InsertNextValue(i)
    
    # create a voxel grid with occupancy scalars
    rgrid = vtk.vtkRectilinearGrid()
    rgrid.SetDimensions(len(x), len(y), len(z))
    rgrid.SetXCoordinates(xCoords)
    rgrid.SetYCoordinates(yCoords)
    rgrid.SetZCoordinates(zCoords)
    rgrid.GetPointData().SetScalars(values)
    
    return rgrid

# function to save grid in various format
def save_grid(rgrid, occupancy, points, threshold, path):
    '''
    Parameters
    ----------
    rgrid : vtk.RectilinearGrid object
        grid containing 3D points with their occupancy value.
    occupancy : np.array
        matrix containing occupancy values.
    points : np.array
        3D voxel points.
    threshold : int
        occupancy threshold for generating mesh object.

    Returns
    -------
    None.
    '''
    pts = points.T
    
    # saving as vtk file with occupancy scalars
    filename = path + "shape.vtk"
    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(rgrid)
    writer.Write()
    
    # saving as ply file after thresholding
    out = pts[occupancy > threshold, :3]
    out_final = [out[i, :] for i in range(out.shape[0])]
    filename = path + "shape.ply"
    vertices = np.array([tuple(e) for e in out_final], dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    el = PlyElement.describe(vertices, 'vertex', {'some_property': 'f8'}, {'some_property': 'u4'})
    PlyData([el], text = True).write(filename)

if __name__ == "__main__":
    # global variables
    image_path = "../data/images/Acrylic/0 degree/"
    tracked_path = "../data/pose/tracked_points/"
    intrinsic_path = "../data/intrinsics/calibration_circle.yml"
    model_path = "../data/model/unet_model"
    projections_path = "../data/pose/drop_Ps.mat"
    mesh_path = "../data/mesh/"
    n_circles = 44
    c_dist = 72
    width = 4
    height = 11
    size = 300
    threshold = 37
    scale_factor = (400, 300, 140)
    (H, W) = (1836, 3264)
    
    # pose estimation
    mtx, dist = load_intrinsics(intrinsic_path)
    newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (W, H), 1, (W, H))
    print("Getting 3D-2D correspondence...")
    objpoints, imgpoints = get_points_circle(image_path, n_circles, c_dist, width, height)
    save_projections_and_pose(objpoints, imgpoints, mtx, dist, newmtx, projections_path)
    
    # delete undetected files
    dstlist = os.listdir(tracked_path)
    deleted_count = 0
    for file in sorted(os.listdir(image_path)):
        if file not in dstlist:
            os.remove(image_path + file)
            deleted_count += 1
    print("Deleted files", deleted_count)
    
    # shape-from-silhouette
    projections, images = load_data(projections_path, image_path)
    images = undistort_images(images, mtx, newmtx, dist)
    silhouette = get_silhouette(images, model_path)
    points = create_voxel(size, scale_factor)
    print("Start carving on", len(silhouette),"images...")
    occupancy = start_carving(projections, silhouette, points)
    rgrid = get_rectilinear_grid(size, points, occupancy)
    print("Carving completed.")
    
    # save point cloud
    print("Saving point cloud...")
    save_grid(rgrid, occupancy, points, threshold, mesh_path)
    print("Point cloud saved.")
