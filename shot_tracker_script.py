import cv2 as cv
import pathlib
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import sys

def setup(filename):

    # initiate video capture
    cap = cv.VideoCapture(filename)
    window = "Window"
    cv.namedWindow(window)


    # initiate array of detected ball centers and counter
    centers = []
    counter = [0]

    # initialize params for simple blob detector
    # parameters optimized for finding the rim

    params = cv.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 25

    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 1000

    params.filterByCircularity = True
    params.minCircularity = 0.45
    params.maxCircularity = 0.55

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.85
    params.maxConvexity = 1

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 0.1

    #initialize simple blob detector itself
    sbd = cv.SimpleBlobDetector_create(params)

    # array for detected rim blobs
    rim_blobs = []


    # holder for rim coordinates and radius
    rim = []

    return (cap, centers, counter, sbd, rim_blobs, window, rim)

def find_rim(hsv, rim_blobs, sbd, frame):

    # manually found rim color values
    rimlower, rimupper = np.array([0, 0, 40]), np.array([15, 140, 140])

    rim_mask = cv.inRange(hsv, rimlower, rimupper)
    rim_filtered = cv.bitwise_and(frame, frame, mask = rim_mask)
    rim_rgb = cv.cvtColor(rim_filtered, cv.COLOR_HSV2RGB)
    rim_grayscl = cv.cvtColor(rim_rgb, cv.COLOR_RGB2GRAY)

    # blurs to make rim's outline bigger
    blurred = cv.GaussianBlur(255 - rim_grayscl, (15, 15), 0)
    ret, binary = cv.threshold(blurred, 250, 255, cv.THRESH_BINARY)

    # finds the rim as a blob
    rim_blobs.append(sbd.detect(255 - binary)[0])
    return

# finds basketball by searching for orange circles
def find_ball(hsv, frame, counter, centers):

    # manually found orange thresholds
    lower, upper = np.array([5, 150, 0]), np.array([15, 255, 255])
    
    mask = cv.inRange(hsv, lower, upper)
    filtered = cv.bitwise_and(frame, frame, mask = mask) 

    rgb = cv.cvtColor(filtered, cv.COLOR_HSV2RGB)
    grayscl = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    circles = cv.HoughCircles(grayscl, cv.HOUGH_GRADIENT, 1.5, 5000, param2 = 20, minRadius = 10)
    if circles is not None:
        counter[0] = 0
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(frame,(i[0],i[1]),i[2],(15,255,255),2)
            # draw the center of the circle
            if len(centers) == 0:
                centers.append(i)
            elif ((centers[-1][0] - i[0])**2 + (centers[-1][1] - i[1])**2 )**.5 > centers[-1][2] * 4:
                cv.circle(frame, (500, 500), 200, (0, 255, 0), 2)
                continue
            else:
                centers.append(i)
    else:
        counter[0] += 1
        if counter[0] == 5:
            centers.clear()
            counter[0] = 0
    return

# determines if a shot is attempted and if the shot is made or missed
def shot_outcome(centers, frame, rim):

    for i in range(len(centers)):
        if i != 0:
            cv.line(frame, centers[i][:2], centers[i-1][:2], (255, 0, 0), 2)
        cv.circle(frame, centers[i][:2], 2, (5, 92, 94), 3)
        if not (centers[i][0] - rim[0])**2 + (centers[i][1] - rim[1])**2 < rim[2]**2 * 1.44:
            if i != 0 and (centers[i-1][0] - rim[0])**2 + (centers[i-1][1] - rim[1])**2 < rim[2]**2* 1.44:
                ang = np.arctan2([centers[i][1] -rim[1]], [centers[i][0] - rim[0]])[0]
                if ang > np.pi/6 and ang < np.pi*5/6:
                    cv.putText(frame, "SHOT MADE", (centers[i][0], centers[i][1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                else:
                    cv.putText(frame, "SHOT MISSED", (centers[i][0], centers[i][1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    return

                


# main function for each frame
def perf_op(name, frame, centers, counter, rim_blobs, frame_counter, sbd, rim, make_video, out):

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # specifies number of frames used to find the rim
    # since it only needs to be found once
    rim_finding_frame = 5

    if len(rim_blobs) < rim_finding_frame:
        find_rim(hsv, rim_blobs, sbd, frame)

    # fills rim array after specified number of frames

    if frame_counter == rim_finding_frame:
        rim_xs = np.array(list(map(lambda x: x.pt[0], rim_blobs)))
        rim_ys = np.array(list(map(lambda x: x.pt[1], rim_blobs)))
        rim_rads = np.array(list(map(lambda x: x.size, rim_blobs)))
        rim.append(int(np.median(rim_xs)))
        rim.append(int(np.median(rim_ys)))
        rim.append(int(np.median(rim_rads)))

    find_ball(hsv, frame, counter, centers)

    shot_outcome(centers, frame, rim)

    cv.imshow(name, frame)

    if make_video:
        out.write(frame)


def main(filename, make_video = False):

    if make_video:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))
    else:
        out = None

    try:
        initials = setup(filename)
    except Exception as e:
        print("error on setup")
        print(e)

    cap = initials[0]
    centers = initials[1]
    counter =  initials[2]
    sbd = initials[3]
    rim_blobs = initials[4]
    name = initials[5]
    rim = initials[6]

    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
        frame_counter += 1

        try:
            perf_op(name, frame, centers, counter, 
                    rim_blobs, frame_counter, sbd, rim, 
                    make_video, out)
        except Exception as e:
            print(e)
            break
            
        if cv.waitKey(1) == ord('q'):
            break
        
    cap.release()
    if make_video:
        out.release()
    cv.destroyAllWindows
    return    

if len(sys.argv) == 1:
    print("Error, please specify file name to be analyzed")
elif len(sys.argv) == 2:
    main(sys.argv[1])
elif sys.argv[2] == "True" or sys.argv[2] == "true":
    main(sys.argv[1], True)
else:
    print("Error")
    

    