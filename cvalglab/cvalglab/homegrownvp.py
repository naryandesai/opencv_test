import cv2
import imutils
import numpy as np
from skimage.measure import ransac, LineModelND
import math
import imagefilters
import random
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from skimage import data
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
 

def vadd(pa,pb):
    pax, pay = pa
    pbx, pby = pb
    return (pax+pbx, pay+pby)

def vsub(pa,pb):
    pax, pay = pa
    pbx, pby = pb
    return (pax-pbx, pay-pby)

def vdot(pa, pb):
    pax, pay = pa
    pbx, pby = pb
    return (pax * pbx) + (pay * pby)

def vmag(pa):
    pax,pay = pa
    return math.sqrt((pax*pax) + (pay*pay))

def vsc(pa,s):
    pax,pay= pa
    return (s*pax, s*pay)

def vdiv(pa,s):
    pax,pay= pa
    return (pax/s, pay/s)

def vnorm(p):
    length = vmag(p)
    length = max(length, 0.0001)
    return vdiv(p, length)

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def line_to_points(line):
    pA = (line[0], line[1])
    pB = (line[2], line[3])
    return (pA,pB)

def get_point_to_line_distance(point, line):
    x1,y1,x2,y2 = line
    px,py = point
    a = px - x1
    b = py - y1
    c = x2 - x1
    d = y2 - y1

    dot = a*c + b*d
    lensq = c*c + d*d

    #p = -1
    #if lensq != 0:
        #p = dot / lensq
    lensq = max(lensq, 0.001)
    p = dot / lensq
    
    # xx = 0.0
    # yy = 0.0
    # if p < 0.0:
    #     xx = x1
    #     yy = y1
    # elif p > 1.0:
    #     xx = x2
    #     yy = y2
    # else:
    xx = x1 + p * c
    yy = y1 + p * d
    
    dx = px - xx
    dy = py - yy
    dist = math.sqrt(dx*dx + dy*dy)

    return dist

def get_lines_intersection_point(lineA, lineB):
    seg1Ax,seg1Ay,seg1Bx,seg1By = lineA
    seg2Ax,seg2Ay,seg2Bx,seg2By = lineB

    intersection = False
    ua = (seg2Bx - seg2Ax) * (seg1Ay - seg2Ay) - (seg2By - seg2Ay) * (seg1Ax - seg2Ax)
    ub = (seg1Bx - seg1Ax) * (seg1Ay - seg2Ay) - (seg1By - seg1Ay) * (seg1Ax - seg2Ax)
    denominator = (seg2By - seg2Ay) * (seg1Bx - seg1Ax) - (seg2Bx - seg2Ax) * (seg1By - seg1Ay)

    epsilon = 0.001
    intersectionPoint = (0.0,0.0)

    if abs(denominator) < epsilon:
        if abs(ua) <= epsilon and abs(ub) < epsilon:
            intersection = True
            intersectionPoint = vdiv(vadd((seg1Ax,seg1Ay),(seg1Bx,seg1By)),2.0)
    else:
        ua /= denominator
        ub /= denominator
        
        if ua >= 0.0 and ua <= 1.0 and ub >=0.0 and ub <= 1.0:
            intersection = True
            ix = seg1Ax + ua * (seg1Bx - seg1Ax)
            iy = seg1Ay + ua * (seg1By - seg1Ay)
            intersectionPoint = (ix,iy)

    return (intersection, intersectionPoint)

def ransac(lines, error_margin, iterations, intersectWindow):
    inliers = []
    iwL, iwT, iwR, iwB = intersectWindow

    if lines is not None and len(lines) >= 2:
        current_iteration = 0
        best_score = 0
        
        while current_iteration < iterations:
            current_iteration += 1
            indexA = 0
            indexB = 0
            intersection = False
            intersectionPoint = (0.0,0.0)
            retries = 0

            while not intersection and retries < len(lines)*10:
                indexA = random.randrange(0, len(lines))
                indexB = random.randrange(0, len(lines))

                lineA = lines[indexA]
                lineB = lines[indexB]

                
                pointsA = line_to_points(lineA)
                dirA = vnorm(vsub(pointsA[1], pointsA[0]))
                pointsB = line_to_points(lineB)
                dirB = vnorm(vsub(pointsB[1], pointsB[0]))
                
                # make sure lines are not the same line or (nearly) parallel
                comparison = abs(vdot(dirA, dirB))

                if indexA != indexB and abs(1.0 - comparison) >= 0.025:
                    intersection, intersectionPoint = get_lines_intersection_point(lineA,lineB)

                # make sure intersection is inside the image window
                if intersection:
                    sx, sy= intersectionPoint
                    if sx < iwT or sy < iwL or sx > iwR or sy > iwB:
                        intersection = False

                retries += 1

            if intersection:
                iteration_inliers = [indexA, indexB]
                iteration_score = 2 

                current_index = 0
                for line in lines:
                    if current_index != indexA and current_index != indexB:
                        distance = get_point_to_line_distance(intersectionPoint, line)
                        if distance < error_margin:
                            iteration_inliers.append(current_index)
                            iteration_score += 1
                    current_index += 1

                if iteration_score > best_score:
                    best_score = iteration_score
                    inliers = iteration_inliers
    return inliers

def intersect(P0,P1):
    """P0 and P1 are NxD arrays defining N lines.
    D is the dimension of the space. This function 
    returns the least squares intersection of the N
    lines from the system given by 
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#In_two_dimensions_2
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    """
    # generate all line direction vectors 
    n = (P1-P0)/np.linalg.norm(P1-P0,axis=1)[:,np.newaxis] # normalized

    # generate the array of all projectors 
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T
    # see fig. 1 

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (projs @ P0[:,:,np.newaxis]).sum(axis=0)

    # solve the least squares problem for the 
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R,q,rcond=None)[0]

    return p

class SegmentVPDetection(object):
    def __init__(self, debugMode=False):
        self._debugMode = debugMode

    def find_forward_vanishing_point(self, original, lowerHalf = False):
        original = imutils.resize(original, width=512)
        dbgImgA = original.copy()
        dbgImgB = original.copy()

        contrast = cv2.convertScaleAbs(original,alpha=1.2)
        wb = imagefilters.white_balance(contrast)
        blf = cv2.bilateralFilter(wb, 11, 75, 75)
        mb = cv2.medianBlur(blf, 5)
        
        inputimg = mb

        img = img_as_float(inputimg[::2, ::2])
        animg = img_as_float(inputimg[::2, ::2])
        animg[:,:] = (0.0,0.0,0.0)

        segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=30) 
        boundaries = mark_boundaries(animg, segments)

        rows, cols, channels = boundaries.shape

        start_rowp = 0.0
        end_rowp = 1.0
        if lowerHalf:
            start_rowp = 0.4
            end_rowp = 0.8

        start_row = math.floor(rows * start_rowp)
        end_row = math.floor(rows * end_rowp)
        if lowerHalf:
            if start_row > 0:
                boundaries[0:start_row:,:,:] = (0,0,0)
            if end_row < rows:
                boundaries[end_row:,:,:] = (0,0,0)

        bimg = img_as_ubyte(imutils.resize(boundaries, width=512))
        gs = cv2.cvtColor(bimg, cv2.COLOR_RGB2GRAY)
        lines = cv2.HoughLines(gs,1,np.pi/180, threshold=100, min_theta=np.pi * 0.3, max_theta=np.pi *0.7 )

        right = (1.0,0.0)

        if not lines is None and len(lines) > 0:
            clines = []
            for line in lines:
                rho, theta = line[0]

                a = np.cos(theta)
                b = np.sin(theta)

                x0 = a*rho
                y0 = b*rho

                x1 = int(x0 + 2000*(-b))
                y1 = int(y0 + 2000*(a))

                x2 = int(x0 - 2000*(-b))
                y2 = int(y0 - 2000*(a))

                cartl = (x1,y1,x2,y2)
                
                clines.append(cartl)
            
            for s in clines:
                x1,y1,x2,y2 = s
                cv2.line(dbgImgB,(x1,y1),(x2,y2),(255,255,255),1)
            
            threshold = max(cols/60.0,5.0)
            left = cols * 0.2
            top = rows * 0.0
            right = cols * 0.8
            bottom = rows * 1.0
            inlier_indices = ransac(clines,threshold, iterations = 1000, intersectWindow = (left,top,right,bottom))

            if len(inlier_indices) > 0:
                inliers = [clines[i] for i in inlier_indices]
                ptsA = []
                ptsB = []
                for line in inliers:
                    x1,y1,x2,y2 = line
                    ptsA.append([x1,y1])

                    ptsB.append([x2,y2])
                    cv2.line(dbgImgA, (x1,y1), (x2,y2), (0,255,255), 1)

                nptsA = np.array(ptsA)
                nptsB = np.array(ptsB)
                vp = intersect(nptsA, nptsB)
            
                cv2.circle(dbgImgA, (int(vp[0]),int(vp[1])), 5, (0,0,255), 5)

        final = cv2.cvtColor(gs, cv2.COLOR_GRAY2BGR)
        return (dbgImgA, dbgImgB, final)


class NietoVPDetection(object):
    def __init__(self, debugMode=False):
        self._debugMode = debugMode
   
    def find_forward_vanishing_point(self, img, lowerHalf = False):
            
            img = cv2.GaussianBlur(img,(5,5),sigmaX=1,sigmaY=1)
            dbgImgA = img.copy()
            dbgImgB = img.copy()

            iHeight, iWidth, channels = img.shape
    
            fimg = None
            start_row = 0.0
            end_row = 1.0
            if lowerHalf:
                start_row = 0.45
                end_row = 0.9

            scanWidth = iWidth//50
            # fimg, pp = imagefilters.BoostedStructuredRoadFilter(img,scanWidth,100,start_row, end_row, True)
            fimg = imagefilters.NietoStructuredRoadFilter(img,scanWidth,35,start_row, end_row, True)

            cv2.line(dbgImgA, (10,10), (10 + int(scanWidth), 10), (255,0,0), 4)

            # todo: get corners! (good points to track) and draw lines between those.
            fimg = cv2.cvtColor(fimg, cv2.COLOR_GRAY2BGR)
            fimg = imutils.auto_canny(fimg)
            # fimg = imutils.auto_canny(blurred)
            
            lines = cv2.HoughLines(fimg,1,np.pi/180, threshold=70, min_theta=np.pi * 0.2, max_theta=np.pi *0.8 )
            

            fimg = cv2.cvtColor(fimg, cv2.COLOR_GRAY2BGR)
            right = (1.0,0.0)

            if not lines is None and len(lines) > 0:
                clines = []
                for line in lines:
                    rho, theta = line[0]

                    a = np.cos(theta)
                    b = np.sin(theta)

                    x0 = a*rho
                    y0 = b*rho

                    x1 = int(x0 + 2000*(-b))
                    y1 = int(y0 + 2000*(a))

                    x2 = int(x0 - 2000*(-b))
                    y2 = int(y0 - 2000*(a))

                    cartl = (x1,y1,x2,y2)
                    
                    clines.append(cartl)
                
                for s in clines:
                    x1,y1,x2,y2 = s
                    cv2.line(dbgImgB,(x1,y1),(x2,y2),(255,255,255),1)
                
                threshold = max(iWidth/60.0,5.0)
                left = iWidth * 0.2
                top = iHeight * 0.0
                right = iWidth * 0.8
                bottom = iHeight * 1.0
                inlier_indices = ransac(clines,threshold, iterations = 1000, intersectWindow = (left,top,right,bottom))

                if len(inlier_indices) > 0:
                    inliers = [clines[i] for i in inlier_indices]
                    ptsA = []
                    ptsB = []
                    for line in inliers:
                        x1,y1,x2,y2 = line
                        ptsA.append([x1,y1])

                        ptsB.append([x2,y2])
                        cv2.line(dbgImgA, (x1,y1), (x2,y2), (0,255,255), 1)

                    nptsA = np.array(ptsA)
                    nptsB = np.array(ptsB)
                    vp = intersect(nptsA, nptsB)
                
                    cv2.circle(dbgImgA, (int(vp[0]),int(vp[1])), 5, (0,0,255), 5)

            return (dbgImgA, dbgImgB, fimg)