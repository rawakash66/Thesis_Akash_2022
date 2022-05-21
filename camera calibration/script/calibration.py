# import libraries
import os
import cv2
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['figure.dpi'] = 600
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('axes', linewidth=2)

# function for calibrating the camera using circle pattern
def calibrate_camera_circle(img_path, n_circles, c_dist, width, height):
    '''
    Parameters
    ----------
    img_path : string
        path to circle images.
    n_circles : int
        number of circles in an image divisible by 11.
    c_dist : int
        horizontal distance between two circles divisible by 2

    Returns
    -------
    calibrated coefficients.
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
    imgx = cv2.imread(img_path + images[0], cv2.IMREAD_UNCHANGED)
    H, W, _ = imgx.shape
    
    for fname in images:
        img = cv2.imread(img_path + fname, cv2.IMREAD_UNCHANGED)
        imgx = img.copy()
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
            cv2.imwrite("../data/tracked_points_circle/" + fname, im_with_keypoints)
    
    print("Number of images used for calibration: ", found)
    print("Start camera calibration.....")
    
    # calibrate camera
    ret, mtx, dist, rvecs, tvecs, stdInt, stdExt, per_view_err = cv2.calibrateCameraExtended(objpoints, imgpoints, imgx.shape[:2][::-1], None, None)
    
    print("Calibration completed.")
    
    # print intrinsics and extrinsics values
    display_estimates(mtx, dist, rvecs, tvecs, stdInt, found)
    
    print("Plotting reprojection errors...")
    print("Total reprojection error = ", ret)
    # plot reprojection errors
    visualize_reprojErr(per_view_err, ret, found)
    
    print("\nPlotting distortion model....")
    visualize_distModel(mtx, dist, imgx.shape[:2])
    
    return mtx, dist, rvecs, tvecs

# function to visualize reprojection errors
def visualize_reprojErr(per_view_err, ret, n_images):
    '''
    Parameters
    ----------
    per_view_err : np.array
        per view reprojection error.
    ret : float
        total reprojection error.
    n_images : int
        number of images.

    Returns
    -------
    None.
    '''
    plt.figure(figsize=(30, 20))
    x = np.arange(1,n_images+1)
    rms_per_view = np.ravel(per_view_err)
    plt.bar(x, rms_per_view, label = "RMSE")
    plt.axhline(y = ret, color='k', linestyle='--', label = "RMSE average", linewidth = 3)
    plt.xlabel("Images", fontsize = 40)
    plt.ylabel("Reprojection error (RMSE)", fontsize = 40)
    plt.gca().xaxis.get_major_ticks()[1].label1.set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=32)
    plt.legend(fontsize=38)
    plt.show()

# function to visualize lens distortion model
def visualize_distModel(mtx, dist, img_shape):
    '''
    Parameters
    ----------
    mtx : np.array
        intrinsic matrix.
    dist : np.arry
        distortion coefficients.
    img_shape : tuple
        image dimension.

    Returns
    -------
    None.
    '''
    # dist = [k1,k2,p1,p2,k3]
    # radial distortion     : k1, k2, k3
    dist = dist.ravel()
    
    if len(dist) < 14:
        dist = np.pad(dist, (0, 14 - len(dist)), 'constant')
    
    # creating meshgrid
    n_steps = 20
    x = np.linspace(0, img_shape[1] - 1, n_steps)
    y = np.linspace(0, img_shape[0] - 1, n_steps)
    u, v = np.meshgrid(x, y)
    
    # solving for ray direction
    n_points = len(u.flatten())
    arr = np.vstack((u.flatten(), v.flatten(), np.ones((1, n_points)))).astype(float)
    xyz, resid, rank, s = np.linalg.lstsq(mtx, arr, rcond = None)
    
    # distortion model formation
    xp = xyz[0, :]/xyz[2, :]
    yp = xyz[1, :]/xyz[2, :]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3
    coef = (1 + dist[0]*r2 + dist[1]*r4 + dist[4]*r6)/(1 + dist[5]*r2 + dist[6]*r4 + dist[7]*r6)
    xpp = xp*coef + 2*dist[2]*(xp*yp) + dist[3]*(r2 + 2*xp**2) + dist[8]*r2 + dist[9]*r4
    ypp = yp*coef + 2*dist[3]*(xp*yp) + dist[2]*(r2 + 2*yp**2) + dist[10]*r2 + dist[11]*r4
    u2 = mtx[0, 0]*xpp + mtx[0, 2]
    v2 = mtx[1, 1]*ypp + mtx[1, 2]
    du = u2[:] - u.flatten()[:]
    dv = v2[:] - v.flatten()[:]
    dr = np.hypot(du, dv)
    dr = dr.reshape(u.shape)
    
    # plot distortion model
    plt.figure()
    ax = plt.subplot(111)
    figure_size = plt.gcf().get_size_inches()
    factor = 3
    plt.gcf().set_size_inches(factor * figure_size)
    plt.quiver(u[:]+1, v[:]+1, du, dv)
    plt.plot(img_shape[1]/2, img_shape[0]/2, 'x', label = 'Ideal principal point location', markersize = 10)
    plt.plot(mtx[0, 2], mtx[1, 2], 'o', label = 'Estimated principal point location', markersize = 10)
    hc = plt.contour(u[0, :]+1, v[:, 0]+1, dr, linewidths = 3)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.01, box.width, box.height * 0.99])
    plt.legend(fontsize = 20, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
    plt.clabel(hc, inline = True, fontsize = 20)
    plt.axis('off')
    plt.show()
    
# function to print intrinsic parameters
def display_estimates(mtx, dist, rvecs, tvecs, stdInt, n_images):
    '''
    Parameters
    ----------
    mtx : np.arry
        intrinsic matrix.
    dist : np.array
        distortion coefficients.
    rvecs : list
        list of rotation vectors.
    tvecs : list
        list of translation vectors.
    stdInt : np.array
        standard deviations.
    n_images : int
        number of images.

    Returns
    -------
    None.
    '''
    dist = np.round(dist.ravel(), 4)
    stdInt = np.round(stdInt.ravel(), 4)
    mtx = np.round(mtx, 4)
    
    if len(dist) < 14:
        dist = np.pad(dist, (0, 14 - len(dist)), 'constant')
    
    if len(stdInt) < 18:
        stdInt = np.pad(stdInt, (0, 18 - len(stdInt)), 'constant')
    
    print('\n')
    print("\t\t\t\tStandard Error of Estimated Camera Parameters")
    print("\t\t\t\t---------------------------------------------")
    print("\n")
    print("Focal length (pixels):    [", mtx[0, 0], '+/-', stdInt[0], '  ', mtx[1, 1], '+/-',stdInt[1], "]")
    print("Principal point (pixels): [", mtx[0, 2], '+/-', stdInt[2], '  ', mtx[1, 2], '+/-', stdInt[3], "]")
    print("Radial distortion:        [", dist[0], '+/-', stdInt[4], '  ', dist[1], '+/-', stdInt[5], '  ', dist[4], '+/-', stdInt[8], "]")
    print("Tangential distortion:    [", dist[2], '+/-', stdInt[6], '  ', dist[3], '+/-', stdInt[7], "]")
    print("\n")    

# function to save coefficients
def save_intrinsics(mtx, dist, path):
    '''
    Parameters
    ----------
    mtx : numpy matrix
        intrinsic camera matrix.
    dist : np.array
        distortion coefficients.
    path : string
        location to store the data.

    Returns
    -------
    None.
    '''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    
    cv_file.release()

if __name__ == "__main__":
    # global parameters
    img_path = "../data/images_circle/"
    n_circles = 44
    c_dist = 72
    width = 4
    height = 11
    (H, W) = (1836, 3264)
    save_path = "../data/calibration_circle.yml"
    
    # calibrate camera
    mtx, dist, rvecs, tvecs = calibrate_camera_circle(img_path, n_circles, c_dist, width, height)
    
    # save intrinsic
    save_intrinsics(mtx, dist, save_path)