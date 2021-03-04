import cv2
result = cv2.imread()


(B, G, R) = cv2.split(result)
B = [i/255 for i in B]
G = [i/255 for i in G]
R = [i/255 for i in R]
# print(R[0])
# print(G[0])
# print(B[0])

for i in range(len(R)):
    for k in range(len(R[i])):
        max_ = max(R[i][k], G[i][k], B[i][k])
        min_ = min(R[i][k], G[i][k], B[i][k])
        h = 0.0
        s = 0.0
        l = 0.0

        if max_ == min_:
            h = 0;
        elif max_ == R[i][k] and G[i][k] > B[i][k]:
            h = 60*(G[i][k] - B[i][k])/(max_ - min_)
        elif max_ == R[i][k] and G[i][k] <= B[i][k]:
            h = 60*(G[i][k] - B[i][k])/(max_ - min_) + 360
        elif max_ == G[i][k]:
            h = 60*(B[i][k] - R[i][k])/(max_ - min_) + 120
        elif max_ == B[i][k]:
            h = 60*(R[i][k] - G[i][k])/(max_ - min_) + 240

        l = (max_ + min_)/2

        if max_ == min_ or l == 0:
            s = 0
        elif 0 < l <= 0.5:
            s = (max_ - min_) / (2 * l)
        elif l > 0.5:
            s =  (max_ - min_) / (2 - 2 * l)