import cv2 as cv 
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import concurrent.futures


data_dir = (os.getcwd()+'\\Data\\')
files_list = os.listdir(data_dir)
output_dir = (os.getcwd()+'\\Output\\')

def write_image(image, output_path):
    if output_path == 'FILTERED RECTANGLES.jpg':
        cv.imwrite(f'({time.time()}) - {output_path}', image)

def to_gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def to_binary(image):
    _, binary = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)
    return binary

def dilate(image, iters,kernel=None):
    return cv.dilate(image, kernel , iterations=iters)

def process_image(image):

    image_gray = to_gray(image)
    binary = to_binary(image_gray)
    dilated = dilate(binary, 2)

    return image_gray, binary, dilated

def load_image(image_path):
    img = cv.imread(image_path)
    image = cv.imread(image_path)

    image_gray, binary, dilated = process_image(image)
    
    write_image(image, 'RGB.jpg')
    write_image(binary, 'BINARY.jpg')
    write_image(dilated, 'DILATED.jpg')

    return image, binary, dilated

def find_contours(contour_image, write_image):
    wimage = write_image.copy()
    contours , _ = cv.findContours(contour_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(wimage, contours, -1, (0, 255, 0), 3)
    return contours, wimage

def rectangular_contours(dilated, image):
    rectangles = []
    
    contours, wimage = find_contours(dilated, image)
    write_image(wimage, 'CONTOURS.jpg')

    rimage = image.copy()

    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
        if len(approx) == 4:
            if cv.isContourConvex(approx):
                cv.drawContours(rimage, [approx], 0, (0, 255, 0), 2)
                rectangles.append(approx)

    write_image(rimage, 'RECTANGULAR CONTOURS.jpg')

    return rectangles

def largest_rectangular_contour(rectangles, image):
    max_area = 0
    max_contour = None
    wimage = image.copy()
    for contour in rectangles:
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    cv.drawContours(wimage, [max_contour], 0, (0, 255, 0), 2)
    write_image(wimage, 'LARGEST CONTOUR.jpg')

    return max_contour

def warp_perspective(image, max_contour):
    points = max_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")

    points_sum = points.sum(axis = 1)
    rect[0] = points[np.argmin(points_sum)] #tleft
    rect[2] = points[np.argmax(points_sum)] #tright

    points_diff = np.diff(points, axis = 1)
    rect[1] = points[np.argmin(points_diff)]#bright
    rect[3] = points[np.argmax(points_diff)]#bleft

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxHeight = max(int(heightA), int(heightB))

    destination = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv.getPerspectiveTransform(rect, destination)
    
    wimage = image.copy()
    warped = cv.warpPerspective(wimage, M, (maxWidth, maxHeight))
    
    write_image(warped, 'PERSPECTIVE TRANSFORM.jpg')

    image_gray, binary, dilated = process_image(warped)

    write_image(binary, 'PERSPECTIVE TRANSFORM BINARY.jpg')
    write_image(dilated, 'PERSPECTIVE TRANSFORM DILATED.jpg')

    return warped, binary, dilated

def extend_line(x1, y1, x2, y2, factor=1.5):
    dx = x2 - x1
    dy = y2 - y1
    x1_new = int(x1 - factor * dx)
    y1_new = int(y1 - factor * dy)
    x2_new = int(x2 + factor * dx)
    y2_new = int(y2 + factor * dy)
    return x1_new, y1_new, x2_new, y2_new

def find_lines(warped_dilated, warped_binary):
    hor = np.array([np.ones(28)])
    ver = np.array(np.ones((28,1)))

    hor_eroded = cv.erode(warped_dilated, hor, iterations=15)
    hor_dilated = cv.dilate(hor_eroded, hor, iterations=15)

    ver_eroded = cv.erode(warped_dilated, ver, iterations=18)
    ver_dilated = cv.dilate(ver_eroded, ver, iterations=15)

    table = cv.add(hor_dilated, ver_dilated)
    write_image(table, 'TABLE.jpg')
    no_lines = cv.subtract(table, warped_binary)
    blank = np.zeros_like(table)
    lines = cv.HoughLines(table, 1, (np.pi)/4/180, 2000)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            # Extend the line
            x1, y1, x2, y2 = extend_line(x1, y1, x2, y2, factor=50)
            
            cv.line(blank, (x1, y1), (x2, y2), (255, 255, 255), 10)

    no_lines = cv.subtract(warped_binary, table)
    write_image(no_lines, 'NO LINES.jpg')
    table = blank
    write_image(blank, 'LINES.jpg')
    
    return table, no_lines



def limit_image_size(warped, rectangles):
    areas = []
    for contour in rectangles:
        area = cv.contourArea(contour)
        if area > 1000:
            areas.append(area)

    areas = np.array(areas)
    print('areas:', len(areas))

    # Calculate the first quartile (Q1) and the third quartile (Q3)
    Q1 = np.percentile(areas, 40)
    Q3 = np.percentile(areas, 60)

    # Compute the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds
    lower_bound = Q1 - 7 * IQR
    upper_bound = Q3 + 7 * IQR

    filtered_rectangles = []
    image = warped.copy()

    for contour in rectangles:
        area = cv.contourArea(contour)
        if area > lower_bound and area < upper_bound:
            filtered_rectangles.append(contour)
            cv.drawContours(image, [contour], 0, (0, 255, 0), 2)

    write_image(image, 'FILTERED RECTANGLES.jpg')

    filtered_rectangles = np.array(filtered_rectangles)
    filtered = filtered_rectangles[::-1]
    filtered = filtered.reshape(-1, 4, 4, 2)
    filtered_rectangles = filtered


    return filtered_rectangles

def make_image(image, filename):
    cv.imwrite(filename, image)

def crop_cells(filtered_rectangles, no_lines, folder_num=0):
    if folder_num == 0:
        folder_num = len(os.listdir(output_dir)) 
    for row in range(len(filtered_rectangles)):
        target_size = (600, 300)
        new_dir = f'{output_dir}\\{folder_num}\\'
        folder_num += 1
        os.makedirs(new_dir, exist_ok=True)
        for i , contour in enumerate(filtered_rectangles[row]):
            x, y, w, h = cv.boundingRect(contour)
            crop_amount = 0
            # Crop the image to the bounding rectangle
            cropped_image = no_lines[y+crop_amount:y+h-crop_amount, x+crop_amount:x+w-crop_amount]
            cropped_image = cv.resize(cropped_image, target_size)

            
            # Save the cropped image to the file system
            filename = os.path.join(new_dir, f'image_{i}.jpg')
            make_image(cropped_image, filename)

def process_file(file, file_num ,is_seq=False):
    image_path = data_dir + file
    image, binary, dilated = load_image(image_path)
    rectangles = rectangular_contours(dilated, image)
    max_contour = largest_rectangular_contour(rectangles, image)
    warped, warped_binary, warped_dilated = warp_perspective(image, max_contour)
    rectangles = rectangular_contours(warped_dilated, warped)
    table, no_lines = find_lines(warped_dilated, warped_binary)
    rectangles = rectangular_contours(table, warped)
    filtered_rectangles = limit_image_size(warped, rectangles)

    if is_seq:
        crop_cells(filtered_rectangles, no_lines)
    else:
        crop_cells(filtered_rectangles, no_lines, (file_num*12)+1)        


def sequential_processing():
    for file in files_list:
        process_file(file, True)

def parallel_processing():
    files_list = os.listdir(data_dir)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file, file_num) for file_num, file in enumerate(files_list)]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")


def main():
    parallel_processing()

if __name__ == "__main__":
    main()