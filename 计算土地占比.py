import cv2
def Calculation(path):
    img = cv2.imread(path,2)
    count = 0
    all = 500 * 600
    for i in range(500):
        for j in range(600):
            if img[i][j]:
                count += 1
    return count/all
    
    
rate = []
Path = ['Data1_reference.tif',
 'Data2_reference.tif',
 'Data3_reference.tif',
 'Data4_reference.tif',
 'Data5_reference.tif',
 'Data6_reference.tif',
 'Data7_reference.tif',
 'Data8_reference.tif',
 'Test1_reference.tif',
 'Test2_reference.tif']
for i in Path:
    rate.append(Calculation("result/"+i))
print(rate)