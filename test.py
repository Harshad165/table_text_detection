import os
import cv2
import pytesseract
import task_classification

# This only works if there's only one table on a page
# Important parameters:
#  - morph_size
#  - min_text_height_limit
#  - max_text_height_limit
#  - cell_threshold
#  - min_columns

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

delta = 0

def pre_process_image(img, save_in_file, morph_size=(10, 2)):

    # get rid of the color
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold
    pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # dilate the text to make it solid spot
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    pre = ~cpy

    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)
    return pre

def show_all_boxes(pre, min_text_height_limit=6, max_text_height_limit=40):
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    temp = pre.copy()
    temp[:,:] = 255

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(temp, (x, y), (x + w, y + h), (0,255,0), 1)

    cv2.imwrite(os.path.join("./data", "boxes.jpg"), temp)

def show_filered_boxes(pre, img, min_text_height_limit=6, max_text_height_limit=40):
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    temp = pre.copy()
    temp[:] = 255

    img = img.copy()
    
    # Getting the texts bounding boxes based on the text size assumptions
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        h = box[3]

        if min_text_height_limit < h < max_text_height_limit:
            boxes.append(box)

    for box in boxes:
        (x,y, w, h) = box
        cv2.rectangle(img, (x-delta, y-delta), (x + w+delta, y + h + delta), (0,255,0), 1)
    
    cv2.imwrite(os.path.join("./data", "filterd_boxes.jpg"), img)
    # return boxes


def find_text_boxes(pre, min_text_height_limit=6, max_text_height_limit=40):
    # Looking for the text spots contours
    # OpenCV 3
    # img, contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 4
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Getting the texts bounding boxes based on the text size assumptions
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        h = box[3]

        if min_text_height_limit < h < max_text_height_limit:
            boxes.append(box)

    return boxes


def find_table_in_boxes(boxes, cell_threshold=10, min_columns=2):
    rows = {}
    cols = {}

    # Clustering the bounding boxes by their positions
    for box in boxes:
        (x, y, w, h) = box
        col_key = x // cell_threshold
        row_key = y // cell_threshold
        cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
        rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

    # Filtering out the clusters having less than 2 cols
    table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    # Sorting the row cells by x coord
    table_cells = [list(sorted(tb)) for tb in table_cells]
    # Sorting rows by the y coord
    table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

    return table_cells


def build_lines(table_cells):
    if table_cells is None or len(table_cells) <= 0:
        return [], []

    max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
    max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

    max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
    max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

    hor_lines = []
    ver_lines = []

    for box in table_cells:
        x = box[0][0]
        y = box[0][1]
        hor_lines.append((x, y, max_x, y))

    for box in table_cells[0]:
        x = box[0]
        y = box[1]
        ver_lines.append((x, y, x, max_y))

    (x, y, w, h) = table_cells[0][-1]
    ver_lines.append((max_x, y, max_x, max_y))
    (x, y, w, h) = table_cells[0][0]
    hor_lines.append((x, max_y, max_x, max_y))

    return hor_lines, ver_lines


def find_table_cells(text_boxes):
    cluster_id = {}
    for idx,box in enumerate(text_boxes):
        cluster_id[box] = -1
        for i in range(0, idx):
            if overlap(box, text_boxes[i]):
                cluster_id[box] = i
                break
        if cluster_id[box] == -1:
            cluster_id[box] = idx
    
    clusters = {}
    maxCols = 0
    for key, val in cluster_id.items():
        if val not in clusters:
            clusters[val] = []
        clusters[val].append(key)
        maxCols = max(maxCols, len(clusters[val]))
    
    ret = []
    for val in clusters.values():
        if len(val) == maxCols:
            ret.append(val)
    for idx,r in enumerate(ret):
        ret[idx] = sorted(r, key=lambda x: x[0])
    ret = sorted(ret, key=lambda x: x[0][1])
    return ret

def draw_boxes(img, rows, path):
    temp = img.copy()
    temp[:,:] = 255

    for row in rows:
        for contour in row:
            (x,y,w,h) = contour
            cv2.rectangle(temp, (x, y), (x + w, y + h), (0,255,0), 4)

    cv2.imwrite(path, temp)

    

def overlap(box1, box2):
    box1_y2 = box1[1] + box1[3]
    box2_y1 = box2[1]
    overlap_y1 = min(box1_y2, box2_y1)
    overlap_y2 = max(box1_y2, box2_y1)
    overlap_len = overlap_y2 - overlap_y1

    if (overlap_y1 == box1_y2):
        return False
    if overlap_len <= 0.25*min(box1[3], box2[3]):
        return False
    return True

def get_text(rows, img):
    im = img.copy()
    pre = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Otsu threshold
    im = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    table_data = {}
    headers = rows[0]
    for idx, header in enumerate(headers):
        (x,y,w,h) = header
        # rect = cv2.rectangle(im, (x-delta, y-delta), (x + w + delta, y + h + delta), (0, 255, 0), 2)
        cropped = im[y-delta:y + delta + h, x-delta:x+delta + w]
        cv2.imwrite(f'./data/header_cropped_{idx}.png', cropped)
        # return
        header_text = pytesseract.image_to_string(cropped, config="--psm 10").strip()
        print("header_text: " + header_text)

        table_data[header_text] = []
        for ridx,row in enumerate(rows[1:]):
            (x,y,w,h) = row[idx]
            # rect = cv2.rectangle(im, (x-delta, y-delta), (x + w + delta, y + h + delta), (0, 255, 0), 2)
            cropped1 = im[y-delta:y + delta + h, x-delta:x+delta + w]
            cropped1 = cv2.resize(cropped1, (cropped1.shape[1]*2, cropped1.shape[0]*2))
            cv2.imwrite(f'./data/header_{ridx}_{idx}_.png', cropped1)
            text = pytesseract.image_to_string(cropped1, config="--psm 10").strip()
            print(idx, text)
            table_data[header_text].append(text)
    print(table_data)

    

                
if __name__ == "__main__":
    path = os.path.join("./data", "inp4.jpg")
    img = cv2.imread(path)
    img = task_classification.task_classification(img)
    # img = img[:int(img.shape[1]*0.8), :]

    pre_file = os.path.join("./data", "pre.png")
    out_file = os.path.join("./data", "out.png")

    pre_processed = pre_process_image(img, pre_file)
    show_all_boxes(pre_processed)
    show_filered_boxes(pre_processed, img)
    text_boxes = find_text_boxes(pre_processed)
    rows = find_table_cells(text_boxes)
    get_text(rows, img)
    # print(table_cells)
    draw_boxes(img, rows, os.path.join("./data", "table_boxes.png"))  
    cells = find_table_in_boxes(text_boxes)
    hor_lines, ver_lines = build_lines(cells)

    # Visualize the result
    vis = img.copy()

    for line in hor_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    for line in ver_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imwrite(out_file, vis)