import cv2
import numpy as np
from skimage.measure import ransac, LineModelND
import math
from numba import jit


def put_text(img, text, org, font_face, font_scale, color, thickness=1, line_type=8, bottom_left_origin=False):
    """Utility for drawing text with line breaks

    :param img: Image.
    :param text: Text string to be drawn.
    :param org: Bottom-left corner of the first line of the text string in the image.
    :param font_face: Font type. One of FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX,
                          FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL,
                          FONT_HERSHEY_SCRIPT_SIMPLEX, or FONT_HERSHEY_SCRIPT_COMPLEX, where each of the font ID’s
                          can be combined with FONT_ITALIC to get the slanted letters.
    :param font_scale: Font scale factor that is multiplied by the font-specific base size.
    :param color: Text color.
    :param thickness: Thickness of the lines used to draw a text.
    :param line_type: Line type. See the line for details.
    :param bottom_left_origin: When true, the image data origin is at the bottom-left corner.
                               Otherwise, it is at the top-left corner.
    :return: None; image is modified in place
    """
    # Break out drawing coords
    x, y = org

    # Break text into list of text lines
    text_lines = text.split('\n')

    # Get height of text lines in pixels (height of all lines is the same)
    _, line_height = cv2.getTextSize('', font_face, font_scale, thickness)[0]
    # Set distance between lines in pixels
    line_gap = line_height // 3

    for i, text_line in enumerate(text_lines):
        # Find total size of text block before this line
        line_y_adjustment = i * (line_gap + line_height)

        # Move text down from original line based on line number
        if not bottom_left_origin:
            line_y = y + line_y_adjustment
        else:
            line_y = y - line_y_adjustment

        # Draw text
        cv2.putText(img,
                    text=text_lines[i],
                    org=(x, line_y),
                    fontFace=font_face,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness,
                    lineType=line_type,
                    bottomLeftOrigin=bottom_left_origin)


def white_balance(img, startrow = 0):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[startrow:, :, 1])
    avg_b = np.average(result[startrow:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def NietoStructuredRoadFilter(image, filter_width, threshold=128, start_row_pct = 0.0, end_row_pct = 1.0, do_threshold = True):
    gsimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retval = np.zeros_like(gsimg)
    inputImage = gsimg.astype(int)
    rows, cols, _ = image.shape
    maxlum = inputImage[:,:].max()
    print(f'n image max: {maxlum}')
       
    start_col = filter_width
    end_col = cols - filter_width

    start_row = math.floor(rows * start_row_pct)
    end_row = math.floor(rows * end_row_pct)
    filter_width = max(filter_width // 2, 4)
    for j in range(start_row, end_row):
        for i in range(start_col,end_col):
            aux = 0
            aux = 2 * inputImage[j, i]
            left = inputImage[j,i-filter_width]
            right = inputImage[j,i+filter_width]

            aux += -left
            aux += -right
            aux += -abs(left - right)

            # force to range
            aux = min(aux, 255)
            aux = max(aux, 0)
            retval[j,i] = aux

    if do_threshold:
        (T,thresh) = cv2.threshold(retval, threshold, 255, cv2.THRESH_BINARY)
        retval = cv2.bitwise_and(retval,retval, mask=thresh)

    return retval

def BoostedStructuredRoadFilter(image, filter_width, threshold=128, start_row_pct = 0.0, end_row_pct = 1.0, do_threshold = True):
        
    rows, cols, channels = image.shape
    start_col = filter_width
    end_col = cols - filter_width

    start_row = math.floor(rows * start_row_pct)
    end_row = math.floor(rows * end_row_pct)


    # blurred = cv2.GaussianBlur(image, (5,5), sigmaX=5, sigmaY=5)
    # wbimg = white_balance(image, start_row)
    # pp = wbimg
    # pp = cv2.convertScaleAbs(wbimg,alpha=1.2)
    # pp = cv2.medianBlur(pp, 7)
    pp = image

    lab = cv2.cvtColor(pp, cv2.COLOR_BGR2LAB)
    inputImage = lab.astype(int)
    maxlum = lab[:,:,0].max()
    print(f'bn image max lum: {maxlum}')

    scanned = boosted_scan(filter_width, inputImage, start_col, end_col, start_row, end_row)
    
    # clip out the outliers
    scannedmax= scanned.max()
    print(f'bn scanned max: {scannedmax}')
    # toobright = scannedmax * 0.618
    # scanned = np.clip(scanned, a_min = 0.0, a_max = toobright)

    scanned *= 255.0 / scannedmax
    retval = scanned.astype(dtype=np.uint8)
    
    #if do_threshold:
        # (T,thresh) = cv2.threshold(retval, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #(T,thresh) = cv2.threshold(retval, threshold, 255, cv2.THRESH_BINARY)
        #retval = cv2.bitwise_and(retval,retval, mask=thresh)

    return retval, pp

@jit(nopython=True)
def linmap(st,end, t):
    return (st*(1-t) + end*t)

@jit(nopython=True)
def compute_relative_lum(raw_value, lmin, lmed, lmax):
    l,a,b = raw_value

    # rescale luminance based on local values
    lmaxmedrange = max(abs(lmax - lmed),0.00001)
    lminmedrange = max(abs(lmed - lmin),0.00001)

    # ladj = max(l - lmed, 0)
    # lpos = ladj/lmaxmedrange
    # lpos = lpos ** 0.22
    # lexp = min(1.0 + lpos,1.333)
    # pxlum = l ** lexp

    ladj = l - lmed
    sgn = math.copysign(1.0,ladj) 
    amf = abs(max(sgn,0.0))
    bmf = 1.0 - amf

    amexp = linmap(1.0, 1.6, (abs(l - lmed)/lmaxmedrange))
    amexp *= amf
    bmexp = linmap(0.77, 1.0, (abs(l - lmin)/lminmedrange)) 
    bmexp *= bmf
    pxlum = (amf * (l ** amexp)) + (bmf * (l ** bmexp))

    if False:
        lrange = max(abs(lmax - lmin),0.001)
        lt = min(max((pxlum - lmin) / lrange, 0),1)
        lum_factor = linmap(0.0, 2.0, lt)
        
        # how neutral (close to 128) are a and b? how close are they together
        deltaA = abs(a - 128.0)
        deltaB = abs(b - 128.0)
        deltaX = abs(deltaA - deltaB)

        # boost yellow - channel b is blue to yellow, 0 .. 255
        yellowAmount = b
        yellowAmount += 2 * deltaX
        yellowAmount -= deltaA
        # yellowAmount -= 80
        yellowAmount = max(0, yellowAmount)
        yellowAmount = yellowAmount * (2.0 + lum_factor)
        

        # boost white. in L*a*b, white means L is high and a and b are near neutral (128 in opencv)
        whiteAmount = pxlum - ((3*deltaA) + (3*deltaB) + (3*deltaX)) 
        whiteAmount -= lmed
        whiteAmount = max(0, whiteAmount)
        whiteAmount = whiteAmount * (0.0 + lum_factor)
        
        
        pxlum += whiteAmount
        pxlum += yellowAmount

    return pxlum 



@jit(nopython=True)
def boosted_scan(filter_width, inputImage, start_col, end_col, start_row, end_row):
    rows, cols, channels = inputImage.shape
    retval = np.zeros((rows, cols),dtype=np.float32)

    filter_width = max(filter_width // 2, 4)
    scan_size = max(int(1.0 * filter_width), 4)

    for j in range(start_row, end_row):
        for i in range(start_col,end_col):
            # scan window
            wL = max(start_col, i - scan_size)
            wT = max(start_row, j - 4)
            wR = min(end_col, i + scan_size)
            wB = min(end_row, j + 4)

            # measure *relative* luminance. 
            # set luminance threshold based on surrounding values
            input_window = inputImage[wT:wB,wL:wR,0]
            lmax = input_window.max() 
            lmed = np.percentile(input_window, 50)
            lmin = input_window.min()
            

            # measure the relative luminance at this pixel
            rindex = i
            raw_value = (inputImage[j, rindex, 0], inputImage[j, rindex, 1], inputImage[j, rindex, 2])
            pxlum = compute_relative_lum(inputImage[j, i], lmin, lmed, lmax)

            # we want to see a high intensity pulse between darkness, penalize for higher intensities to the side
            rindex = max(i - filter_width, start_col)
            raw_value = (inputImage[j, rindex, 0], inputImage[j, rindex, 1], inputImage[j, rindex, 2])
            leftIntensity = compute_relative_lum(raw_value, lmin, lmed, lmax) 

            rindex = min(i + filter_width, end_col)
            raw_value = (inputImage[j, rindex, 0], inputImage[j, rindex, 1], inputImage[j, rindex, 2])
            rightIntensity =  compute_relative_lum(raw_value, lmin, lmed, lmax)

            rel = 0
            rel = 2 * pxlum
            rel += -leftIntensity
            rel += -rightIntensity

            # penalize if intensities to sides are uneven
            rel += -abs(leftIntensity - rightIntensity)
            rel = max(0, rel)
                   

            # measure using absolute luminance at this pixel
            rindex = i
            pxlum = inputImage[j, rindex, 0]
            rindex = max(i - filter_width, start_col)
            leftIntensity = inputImage[j, rindex, 0]
            rindex = min(i + filter_width, end_col)
            rightIntensity = inputImage[j, rindex, 0]
            
            absl = 0
            absl = 2 * pxlum
            absl += -leftIntensity
            absl += -rightIntensity
            absl += -abs(leftIntensity - rightIntensity)
            absl = max(0,absl)
            absl = min(absl, 255) * 30.0         
            
            # score = (2.0 * rel) + absl 
            # score = max(rel,absl)
            # score = rel + absl
            score = rel
            # score = absl

            retval[j,i] = score

    return retval


