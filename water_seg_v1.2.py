# Imports
import cv2
import numpy as np
import time
import pandas as pd # CHANGED:
from imutils.video import count_frames
import os

# Declaration of which video file is being fed in
ship = 'meters_001'

# Path to file
path = f'/Users/blank/AutoDraft/imagery/{ship}.mp4'
total_frames = count_frames(path)
#print(total_frames)
# Choose to save the output or not
save = True

# CHANGED:
# Bring in average location based across 30 fps
avg_draft_location = pd.read_csv('/Users/blank/AutoDraft/scripts/Average_x_y_positions_30fps.csv')
avg_draft_location = np.array(avg_draft_location)
#print(avg_draft_location)

def nothing(x):
    pass
# filters area of found contours
def area_filter(contours):
    #run each 
    areas = np.array(list(map(cv2.contourArea, contours)))
    # take the mean of these areas
    Ea = np.mean(areas)
    # find the standard deviation of the areas
    Sa = np.std(areas)
    # return a filtered list of the contours that meet the area requirements
    # to pass, each contour grouping area must be larger than 4 stds less
    # than the mean and less than 4 stds greater than the mean.
    return filter(lambda x: Ea-Sa*4 < cv2.contourArea(x) < Ea + Sa*4, contours)

# filters out erroneous lines
def line_filter(ly,ry):

    def _line_filter(box):
        # takes boundingRect output shown below
        x,y,w,h = box
        # Given a set of points, fits a line to minimize the distance to the line and the points
        # QUESTION: What exactly does this do? What line is it building? @ask
        [_,vy1,_,_] = cv2.fitLine(np.array([(0,ly+h*2),(x,y)]), cv2.DIST_HUBER,0,0.01,0.01)
        [_,vy2,_,_] = cv2.fitLine(np.array([(x,y),(frame_width-1,ry+h*2)]), cv2.DIST_HUBER,0,0.01,0.01)
        # only returns values that fall within this range
        return vy1<=0 and vy2>=0
    return _line_filter

# filters out erroneous points
# QUESTION: is this based on size of point?? Confused about what exactly is being filtered @ask
def point_filter(xm, vel): 
    def _point_filter(pt):
        width = min(frame_width//4, frame_width//4*vel*10)
        return 40 < pt[0][1] < frame_height-60 and xm-width < pt[0][0] < xm+width
    return _point_filter

# finds the center point of the filtered boxes
def box_center(box):
    x,y,w,h = box
    return x + w//2, y + h//2

def get_points(box):
    x,y,w,h = box
    return x, y, x + w, y + h

#draws boxes on the frame around draft marks
def draw_box(frame):
    def _draw(box):
        x,y,w,h = box
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

        return box
    return _draw

# Finding white marks within the image
def find_white_marks(frame):
    # convert frame to gray scale
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # retrieve i_lower value from trackbar position
    i_lower = cv2.getTrackbarPos("i lower", controlwindow)
    # use binary thresholding and outputs just the thresholded image. 
    # In this case anything above i_lower becomes white (255), while
    # anything below is given a value of black (0)
    im = cv2.threshold(im, i_lower, 255, cv2.THRESH_BINARY)[1]
    # build a structuring element for image processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # pass image with the structuring element to open --> this is
    # erosion followed by dilation (erosion reduces noise, however 
    # also reduces the size, thus we need to dilate back)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    # this portion of code grabs the contours of our thresholded image
    contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL,
     cv2.CHAIN_APPROX_NONE)[-2:] 
    # iterates the countours ouput from our filter through the boundingRect
    # function and outputs the map_object to boxes
    boxes = map(cv2.boundingRect, area_filter(contours))
    # returns boxes and thresholded/operated image
    #CHANGED: returning contours as well
    #TODO: save contours into some sort of dictionary or array or something to examine it and how to incorporate into labelme
    return boxes, im, contours

# filters out center points that do not fall within a range
def remove_outliers(centers):
    # takes centers of boxes
    centers = np.array(list(centers))
    if len(centers)==0:
        return centers
    # Ex, Ey = centers.mean(0)
    x,y = centers[:,0], centers[:,1]
    x.sort()
    y.sort()
    # Ex = (x[len(centers)//2] + np.mean(x))/2
    Ex = x[len(centers)//2] 
    Ey = y[len(centers)//2]
    Sx, Sy = centers.std(0)
    # filters out centers that do not fall within a range
    # QUESTION: Why does this only filter pased on x-values and not the y-values? @ask
    return filter(lambda pt: Ex-Sx/3 < pt[0] < Ex + Sx/3, centers)

# finds the vertical line that decends down the center of the draft marks
def find_draft_vert(boxes, topx, botx):
    centers = np.array(list(remove_outliers(map(box_center, boxes))))
    if len(centers) == 0:
        return topx, botx
    # outputs the vector and x, y points of the line
    [vx,vy,x,y] = cv2.fitLine(centers, cv2.DIST_HUBER,0,0.01,0.01)
    if vy == 0:
        vy = 0.01
    # finds the top and bottom x-value (which is the height in this case)
    topx = int((-y*vx/vy) + x)
    botx = int(((frame_height-y)*vx/vy)+x)
    # finds min between points on line and frame_width
    # then returns either 0, or the previously found min value, whichever is larger
    return max(0,min(topx, frame_width)), max(0,min(botx, frame_width))

# finds waterline based on input frame 
def find_waterline(frame, lefty, righty, labels, xm, vel):
    # number of colors
    n_colors = 3
    # pos = int(np.min([lefty, righty, frame_height*0.4]))
    # takes draft_pos from trackbar
    draft_pos = cv2.getTrackbarPos("draft_pos", controlwindow)
    # finds actual initial guess position from that
    pos = int(frame_height*draft_pos/255)
    # reshapes the pixel value at the frame where the draft position is at
    pixels = np.float32(frame[pos:].reshape(-1, 3))
    # color set to blue
    color = (255, 0, 0)
    # sets the desired exit criteria base on max iterations and accuracy
    # QUESTION: why is the 200 and .1, in that order? 200 is for max_iter and .1 is EPS yes? Or no? @ask
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    # k-means random clustering set to flags
    flags = cv2.KMEANS_RANDOM_CENTERS
    # outputs labels and the palette (centers based on "flags")
    # QUESTION: does this 3 times, so 600 iterations?
    _, labels, palette = cv2.kmeans(pixels, n_colors, labels, criteria, 3, flags)
    # returns the sorted counts of the unique labels
    _, counts = np.unique(labels, return_counts=True)
    # dominant label is the indece of pallete that has the max count from kmeans labels
    dominant = palette[np.argmax(counts)]
    # lower values for what is still considered that object
    lower = np.array(dominant*0.25, dtype="uint8")
    # upper value of what is still considered that object
    upper = np.array(dominant*1.55, dtype="uint8")
    # creates a mask that is the section of the frame that falls within the upper
    # and lower bound
    mask = cv2.inRange(frame, lower, upper)
    # kernel is the structuring element of an ellipse
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
    # fill in holes in the mask based on the ellipse kernel, runs twice
    # FLAG: Would like further explanation [how exactly does this work?]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # finds the contours of the mask (this should be the waterline)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:] ##########
    # cntr_img = cv2.drawContours(frame, contours, -1, color)

    if len(contours)>0:
        max_idx = np.argmax([cv2.contourArea(cnt) for cnt in contours])
        cnt = contours[max_idx]
        pts = np.array(list(filter(point_filter(xm, vel),cnt)))
        if len(pts) > 0:
            # puts blue points where it estimates the waterline to be
            [cv2.circle(frame,tuple(*pt), 1, (255,0,0), -1) for pt in pts]
            # fits a line to these points
            [vx,vy,x,y] = cv2.fitLine(pts, cv2.DIST_L2,0,0.01,0.01)
            lefty = int((-x*vy/vx) + y)
            righty = int(((frame_width-x)*vy/vx)+y)
            # returns the furthest right and left points of the line
            # FLAG: just want to verify this is correct @ask
    return max(0,min(lefty, frame_height)), max(0,min(righty, frame_height)), labels

# Implement kalman filter for line estimation
# simple example: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
def line_estimator(noise_factor):
    kalman = cv2.KalmanFilter(6, 4)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0 , 0],
                                         [0, 0, 1, 0, 0 , 0],
                                         [0, 0, 0, 1, 0 , 0],
                                         ], np.float32)

    kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0, 0],
                                        [0, 1, 0, 1 ,0 ,0],
                                        [0, 0, 1, 0, 1, 0],
                                        [0, 0, 0, 1, 0 ,1],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0 ,1]], np.float32)

    kalman.processNoiseCov = np.array([[1, 0, 0, 0, 0, 0],
                                       [0, 1, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]], np.float32) * noise_factor
    return kalman

# Line Intersection found utilizing determinants
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

######################### Building the Window ##########################
# Video window Title
controlwindow = "Draft Estimate"
# Create named window to house video. WINDOW_NORMAL enables manual resizing
cv2.namedWindow(controlwindow, cv2.WINDOW_NORMAL)
# Initial position of the "i lower" track bar
i_lower = 180
# Creates "i lower" trackbar. Pointer begins at position i_lower, and moves to a maximum position of 255.
cv2.createTrackbar("i lower", controlwindow, i_lower, 255, nothing)
# Initial position of "draft pos" trackbar
draft_pos = 170
# Creates "draft pos" trackbar. Pointer begins at draft_pos, and moves to maximum position of 255.
cv2.createTrackbar("draft_pos", controlwindow, draft_pos, 255, nothing)

######################### The Stuff ###################################################
# Grab video from path and save to variable
cap = cv2.VideoCapture(path)
# Get frame width and height using VideoCapture.get(propID) with propID as specified
frame_width = int(cap.get(3))
print(frame_width)
frame_height = int(cap.get(4))
print(frame_height)
# Area of Frame
area = frame_width*frame_height

# Write the video out to file with the below specifications if save=TRUE
# Writes videos based on the following specs: name, fourcc video codec id, fps, frame dimensions
if save:
    out = cv2.VideoWriter(f'{ship}_pred_draft.avi', 
        cv2.VideoWriter_fourcc('M','J','P','G'),
         60, (frame_width,frame_height))
ret = True

# Establish left and right hand as well as top and bottom portions of the frame (// division notation results in type int)
# The height is the left and right portions of the frame, whereas the width is the top and bottom in OpenCV,
# hence the notation below of "lefty" and "topx"
lefty = righty = frame_height//2
topx = botx = frame_width//2
# take above points to create a line and then find their intersection
xm, ym = line_intersection(((topx,0), (botx,frame_height-1)), 
    ((0, lefty-1), (frame_width-1, righty)))

# Create timer
timer = []
it = 0
n = 0 # established for averaging
# set up empty arrays for filling with draft points
x_draft_points = np.empty(total_frames)
y_draft_points = np.empty(total_frames)
wl_k = line_estimator(0.3)
d_k = line_estimator(0.3)
c_k = line_estimator(0.3)

old_im = np.zeros([frame_height, frame_width])
hsv = np.zeros([frame_height, frame_width, 3], "uint8")
hsv[...,1] = 255
vxm = vym = 0.0
labels = None
# start timer returning time in seconds since the epoch as a float
start = time.time()
while(ret):
    # cap.read returns the current frame value, and the image
    ret, frame = cap.read()
    fps = (it+1)/((time.time() - start))
    # print(fps) CHANGED: commented out printing FPS
    # if there is a frame available
    if frame is not None:
        # copy frame
        _frame = frame.copy()
      
        boxes, im, contours = find_white_marks(frame)

        # calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(old_im,im, 
            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # finds the flow in different sections of the frame
        tfvx, _ = flow[:frame_height//2,:,:].mean((0,1))
        bfvx, _ = flow[frame_height//2:,:,:].mean((0,1))
        _, lfvy = flow[:,:frame_width//2,:].mean((0,1))
        _, rfvy = flow[:,frame_width//2:,:].mean((0,1))
        fvx, fvy = flow.mean((0,1))
        m_s = np.sqrt(vxm**2 + vym**2)
        # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # hsv[...,0] = ang*180/np.pi/2
        # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        # flowim = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        # draws boxes around draft markings (following filtering)
        # CHANGED: This section here allows for the saving of the filtered boxes for import into a JSON file @labelme @JSON
        boxes_for_labelme = map(cv2.boundingRect, area_filter(contours))
        boxes_for_labelme = list(map(draw_box(frame), filter(line_filter(lefty,righty), 
            boxes_for_labelme)))
        #TODO: build little function to call that will change these into the actual four points that we want vice, just the values
        print(boxes_for_labelme)

        boxes = map(draw_box(frame), filter(line_filter(lefty,righty), 
            boxes))

        # finds the vertical line that bisects draft marks
        topx, botx = find_draft_vert(boxes, topx, botx)

        # send line through kahlman filter
        d_k.correct(np.array([topx, botx, tfvx,bfvx], np.float32))
        prediction = d_k.predict()
        # creates new values based on the prediction
        topx, botx, tvx, bvx, _, _ = prediction
        # QUESTION: What are we finiding the distnace between here? @ask
        v_s = np.sqrt(tvx**2 + bvx**2)
        # if the above is within a threshold, print the white vertical line on the frame.
        if v_s < 2:
            cv2.line(frame,(int(botx),frame_height-1),(int(topx),0),
                (255,255,255),2)

        # find waterline
        lefty, righty, labels = find_waterline(frame, lefty, righty, labels, xm, m_s)
        # adjust with kalman filter
        wl_k.correct(np.array([lefty, righty, lfvy, rfvy], np.float32))
        prediction = wl_k.predict()
        lefty, righty, lvy, rvy, _, _ = map(int,prediction)
        d_s = np.sqrt(lvy**2 + rvy**2)
        if d_s < 2:
            cv2.line(frame,(frame_width-1,righty),(0,lefty),(0,0,255),2)
        # creates red outline around the frame indicating not sure
        if d_s < 20 and v_s < 20:
            xm, ym = line_intersection(((topx,0), (botx,frame_height-1)),
             ((0, lefty-1), (frame_width-1, righty)))
            c_k.correct(np.array([xm,ym, fvx, fvy], np.float32))
        xm, ym, vxm, vym, _, _ = c_k.predict()
        # QUESTION: what does m_s indicate?? Optical flow rate? @ask
        if m_s < 0.5:
            # creates green rectangle around frame and green circle around point if sure
            cv2.rectangle(frame, (0,0), (frame_width-1, frame_height-1),
             (0, 255, 0), 14)
            #cv2.circle(frame, (int(xm),int(ym)), 14, (0, 255, 0), 10) # CHANGED: Commented out larger circle
            # print(xm, ym)
            # fill x and y draft point arrays
            #x_draft_points[it] = int(xm)
            #y_draft_points[it] = int(ym)
            # print(x_draft_points, y_draft_points)
        else:
            # create red box around frame indicating not sure
            cv2.rectangle(frame, (0,0), (frame_width-1, frame_height-1),
             (0, 0, 255), 14)
        # add point indicating where draft is
        # CHANGED: added in this whole section to produce the average draft position per second (it updates every 30 frames)
        if it < 1043:
            #print("n =", n)
            if it == 0:
                #print(data[0, :])
                cv2.circle(frame, (avg_draft_location[0,1], avg_draft_location[0,2]), 2, (0, 0, 0), 2)
                #print("Loop 1: it == 0")
                #print(avg_draft_location[0,1], avg_draft_location[0,2])
            elif n > 32:
                #print(data[33, :])
                cv2.circle(frame, (avg_draft_location[33,1], avg_draft_location[33,2]), 2, (0, 0, 0), 2)
                #print("Loop 2: n > 32")
                #print(avg_draft_location[33,1], avg_draft_location[33,2])
            elif it % 30 != 0:
                #print(data[n, :])
                cv2.circle(frame, (avg_draft_location[n,1], avg_draft_location[n,2]), 2, (0, 0, 0), 2)
                #print("Loop 3: % 30 != 0")
                #print(avg_draft_location[n,1], avg_draft_location[n,2])
            else:
                n+=1
                #print(data[n, :])
                cv2.circle(frame, (avg_draft_location[n,1], avg_draft_location[n,2]), 2, (0, 0, 0), 2)
                #print("Loop 4: % 30 == 0")

        # show the images
        # cv2.imshow("Draft Marks Estimate", np.concatenate([frame, flowim],1))
        alpha = 0.4  # Transparency factor.

        # Following line overlays transparent rectangle over the image
        frame = cv2.addWeighted(_frame, alpha, frame, 1 - alpha, 0)
        cv2.imshow(controlwindow, frame)
        #TODO write frame to jpg
        labelme_path = '/Users/blank/AutoDraft/imagery/labelme_playground/meters_001_output'
        cv2.imwrite(os.path.join(labelme_path, f'frame_{it}_{ship}.jpg'), frame)
        it+=1
        #print(it)
    if save:
        out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# find average pixel location of draft

#x_y_points = np.column_stack(x_draft_points, y_draft_points)
#np.savetxt("AutoDraft_draft_points.csv", x_y_points, delimiter=",")
#average_x_draft_point = int(np.mean(x_draft_points))
#average_y_draft_point = int(np.mean(y_draft_points))
#print(x_draft_points, y_draft_points)

if save:
    out.release()
cap.release()
cv2.destroyAllWindows()


