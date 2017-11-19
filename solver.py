import cv2
import numpy as np
import sudoku

def get_clf(trainfile,labelfile):
    samples = np.float32(np.loadtxt(trainfile))
    responses = np.float32(np.loadtxt(labelfile))

    clf = cv2.ml.KNearest_create()
    clf.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return clf


def transform_warp_img(sud_cnt,img,thresh):
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    cv2.drawContours(mask, [sud_cnt], -1, 255, -1)
    im = thresh.copy()
    im[mask == 0] = 0
    peri = cv2.arcLength(sud_cnt, True)
    approx = cv2.approxPolyDP(sud_cnt, 0.02 * peri, True)
    points = approx.reshape(4, 2)
    order = np.zeros((4, 2), dtype=np.float32)
    s = np.sum(points, axis=1)
    order[0] = points[np.argmin(s)]
    order[2] = points[np.argmax(s)]
    d = np.diff(points, axis=1)
    order[1] = points[np.argmin(d)]
    order[3] = points[np.argmax(d)]

    w1 = np.sqrt((order[1][0] - order[0][0]) ** 2 + (order[1][1] - order[0][1]) ** 2)
    w2 = np.sqrt((order[2][0] - order[3][0]) ** 2 + (order[2][1] - order[3][1]) ** 2)

    h1 = np.sqrt((order[0][0] - order[3][0]) ** 2 + (order[0][1] - order[3][1]) ** 2)
    h2 = np.sqrt((order[1][0] - order[2][0]) ** 2 + (order[1][1] - order[2][1]) ** 2)

    finw = max(int(w1), int(w2))
    finh = max(int(h1), int(h2))
    finp = np.array([[0, 0], [finw, 0], [finw, finh], [0, finh]], dtype=np.float32)

    trans = cv2.getPerspectiveTransform(src=order, dst=finp)
    warp = cv2.warpPerspective(img, trans, (finw, finh))
    warp = cv2.resize(warp, (450, 450))
    return warp

def get_roi(testfile):
    img = cv2.imread(testfile)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    _,cnts,_ = cv2.findContours(thresh,cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)

    sud_area = 0
    sud_cnt = None

    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if(len(approx)==4):
            x,y,w,h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if(0.7 < float(w)/h < 1.3 and area > 150*150 and area > sud_area and area > 0.5 *w * h):
                sud_area = area
                sud_cnt = cnt

    if sud_cnt is not None:
        warp = transform_warp_img(sud_cnt,img,thresh)
        return warp
    else:
        return None


def process_and_get_sud(clf,warp):
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    _,num_cnt,_ = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    grid = np.zeros((9, 9), dtype=np.int)

    for i in range(9):
        for j in range(9):
            cell = gray[i*50:(i+1)*50][:,j*50:(j+1)*50]
            cell = cv2.GaussianBlur(cell,(5,5),2)
            _,cell_cnt,_ = cv2.findContours(cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cellc in cell_cnt:
                area = cv2.contourArea(cellc)
                (bx, by, bw, bh) = cv2.boundingRect(cellc)
                if(30 < area and 10< bw<45 and 15< bh<45):
                    num_img = cell[by:by+bh][:,bx:bx+bw]
                    num_img = cv2.resize(num_img, (28, 28), interpolation=cv2.INTER_AREA)
                    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                    erode = cv2.erode(num_img, kernel, iterations=1)
                    num_img = cv2.dilate(erode,kernel,iterations =1)
                    image = num_img.reshape((1,784)).astype(np.float32)
                    num = clf.predict(image)
                    grid[i][j] = num[0]

    sudstr = ''.join(str(i) for i in grid.flatten())
    return sudstr

def save_sol(sud_sol,sudstr,warp,solfile):
    for i in range(81):
        if sudstr[i] == '0':
            x = (i%9)*50 + 17
            y = int(i/9)*50 + 40
            cv2.putText(warp, sud_sol[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (181, 2, 1), 2)
    cv2.imwrite(solfile,warp)

if __name__ == '__main__':
    trainfile = r'data/train.data'
    labelfile = r'data/label.data'
    testfile = r'data/test/test3.jpg'
    solfile = r'output/t3_ans.jpg'
    clf = get_clf(trainfile,labelfile)
    warp = get_roi(testfile)
    if warp is not None:
        sudstr = process_and_get_sud(clf,warp)
        sud_sol = sudoku.solve_sudoku(sudstr)
        save_sol(sud_sol,sudstr,warp,solfile)
    else:
        print('Failed to find Puzzle!')

