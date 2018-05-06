import cv2
import numpy as np

def get_angle(p1, p2, p3, p4):
    g0 = p2-p1
    g1 = p4-p3

    return np.math.atan2(np.linalg.det([g0,g1]),np.dot(g0,g1))

def calc_angle_in_degrees(v1, v2):
    p1 = np.array([v1[0], v1[1]])
    p2 = np.array([v1[2], v1[3]])
    p3 = np.array([v2[0], v2[1]])
    p4 = np.array([v2[2], v2[3]])
    return np.degrees(get_angle(p1,p2,p3,p4))


## solution of NetEase - JI of 0.8820 and CI [0.8790, 0.8850]
# Their method starts by extracting line segments by the LSD method,
# such segments are then grouped and quadrangles are formed by selecting
# two horizontal and vertical segment groups. The final quadrangle is
# selected based on its aspect-ratio, area and inner angles.
if __name__ == "__main__":
    # read grayscale image
    #load testimage - TODO: load whole dataset
    img = cv2.imread("data/images/all_objects/background01-letter005-frame112.jpg",0) #0 = grayscale

    #down-sampling? - TODO: proof!

    #apply canny edge - optional? - TODO: proof!
    #img = cv2.Canny(img,100,200)
    #create LSD detector with standard or no refinement
    #lsd = cv2.createLSDDetector(0)
    lsd = cv2.createLineSegmentDetector(0)


    #detect the lines
    lines,width,prec,nfa = lsd.detect(img)
    linesHorzontal = []
    linesVertical = []

    #groupe segments - lines is vec4i
    for i in range(0,lines.size//4):
        if abs(lines[i][0][0]-lines[i][0][2])<abs(lines[i][0][1]-lines[i][0][3]):
            linesHorzontal.append(lines[i][0])
        else:
            linesVertical.append(lines[i][0])

    linesHorzontal = np.array(linesHorzontal)
    linesVertical = np.array(linesVertical)

    #form quadrangles = selecting 2 horizontal and 2 vertical segment groups
    quadranglesHor = []
    quadranglesVer = []

    for i in range(0,linesHorzontal.size//4):
        for n in range(i+1,linesHorzontal.size//4):
            quadranglesHor.append([linesHorzontal[i],linesHorzontal[n]])

    for i in range(0,linesVertical.size//4):
        for n in range(i+1,linesVertical.size//4):
            quadranglesVer.append([linesVertical[i],linesVertical[n]])

    quadranglesHor = np.array(quadranglesHor)
    quadranglesVer = np.array(quadranglesVer)
    # print(quadrangles[0][0])
    # print(quadrangles[0][1])
    # print(quadrangles[5][0])
    # print(quadrangles[5][1])

    #choose final quadrangle (=quadrilaterals) = based on aspect-ration, area and inner angles
    # interior angeles 360 degrees
    v_ver1 = quadranglesVer[0,0]
    v_ver2 = quadranglesVer[0,1]

    v_hor1 = quadranglesHor[0,0]
    v_hor2 = quadranglesHor[0,1]

    # calculate inner angle
    angle0 = calc_angle_in_degrees(v_ver1,v_hor1)
    angle1 = calc_angle_in_degrees(v_hor1,v_ver2)
    angle2 = calc_angle_in_degrees(v_ver2,v_hor2)
    angle3 = calc_angle_in_degrees(v_hor2,v_ver1)

    sum = angle0+angle1+angle2+angle3

    print(sum)
    #a = c4d.utils.VectorAngle(v1, v2)

    # area: 1/2 * ad * cd * sin(alpha) + 1/2 * ab * bc * sin(beta)
    print(lines.size)
    print(lines[401:1291,:,:])

    #show found lines
    draw_img = lsd.drawSegments(img,lines[:,:,:])
    #draw_img = lsd.drawSegments(img, lines)

    cv2.imshow("LSD", draw_img)
    cv2.waitKey(0)






    print();





