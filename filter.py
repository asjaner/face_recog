import face_recognition
import cv2

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
specs_ori = cv2.imread('harry.png', -1)
#cigar_ori = cv2.imread('cigar.png',-1)
 
cap = cv2.VideoCapture(0) #webcame video
# cap = cv2.VideoCapture('jj.mp4') #any Video file also
cap.set(cv2.CAP_PROP_FPS, 30)
 
class Face_rec:

    def __init__(self,specs_ori):
        self.specs_ori = [specs_ori]

    def transparentOverlay(self,src,overlay, pos = (0,0), scale = 1):
        self.overlay = cv2.resize(overlay, (0, 0), fx = scale, fy = scale)
        h, w, _ = self.overlay.shape
        rows, cols, _ = src.shape
        y, x = pos[0], pos[1]

        for i in range(h):
            for j in range(w):
                if x + i >= rows or y + j >= cols:
                    continue
                alpha = float(self.overlay[i][j][3] / 255.0)
                src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
        return src
    def get(self,x):
        return self.specs_ori[x]

 
fac = Face_rec(specs_ori)
while 1:
#    ret, img = cap.read()
#    rgb_frame = img[:,:,::-1]
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    faces = face_cascade.detectMultiScale(img, 1.2, 5, 0, (120, 120), (350, 350))
    ret, img = cap.read()
    rgb_frame = img[:,:,::-1]
    face_loc = face_recognition.face_locations(rgb_frame)
    faces = [(0,0,0,0)]
    if face_loc != []:
        faces = [[int(face_loc[0][3]*3/8),int(face_loc[0][0]*8/7),int(abs(face_loc[0][3]-face_loc[0][1])*2.6),int(abs(face_loc[0][0]-face_loc[0][2]))]]
    
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
 
            glass_symin = int(y - 8 * h / 7)
            glass_symax = int(y + 9 * h / 6)
            sh_glass = glass_symax - glass_symin
 
#            cigar_symin = int(y + 4 * h / 6)
#            cigar_symax = int(y + 6 * h / 6)
#            sh_cigar = cigar_symax - cigar_symin
 
            face_glass_roi_color = img[glass_symin:glass_symax, x:x*2+w]
#            face_cigar_roi_color = img[cigar_symin:cigar_symax, x:x+w]
 
            specs = cv2.resize(fac.get(0), (w, sh_glass),interpolation=cv2.INTER_CUBIC)
#            cigar = cv2.resize(fac.get(0), (w, sh_cigar),interpolation=cv2.INTER_CUBIC)
            fac.transparentOverlay(face_glass_roi_color,specs)
#            fac.transparentOverlay(face_cigar_roi_color,cigar,(0,0),0.6)
 
    cv2.imshow('Thugs Life', img)
 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imwrite('img.jpg', img)
        break
 
cap.release()
 
cv2.destroyAllWindows()
