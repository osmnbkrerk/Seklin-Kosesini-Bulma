import cv2
import numpy as np

resim = cv2.imread('ucgen.png')
griResim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(griResim, 150, 255, cv2.THRESH_BINARY)
konturler, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for i in konturler:
    resim = cv2.imread('ucgen.png')
    griResim = np.float32(griResim)
    mask = np.zeros(griResim.shape, dtype="uint8") #veri tipi 8 bitlik unsigned int olan griResimin shape'i kadar 0 oluşturup dizi yaratır
    cv2.fillPoly(mask, [i], (255, 255, 255))
    dst = cv2.cornerHarris(mask, 5, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, etiketler, istatistikler, geo_merkez = cv2.connectedComponentsWithStats(dst)
    kriter = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    koseler = cv2.cornerSubPix(griResim, np.float32(geo_merkez), (5, 5), (-1, -1), kriter)


if len(koseler) == 4:
    print('Resimdeki şekil üçgendir. \nÜçgenin Köşeleri: ')

    for i in range(1, len(koseler)):
        print(koseler[i]) #her bir köşenin x,y koordinatlarını yazdırıyoruz

        cv2.circle(resim, tuple(np.round(koseler[i]).astype("int")), 3, (255, 0, 0), 4)
        #circle metodunda center parametresine integer elemanlı tuple göndermem gerekiyordu. böyle bir dönüşüm yapmam gerekti
else:
    print("Resimdeki şekil üçgen değildir.")
    for i in range(1, len(koseler)):
        print(koseler[i]) #her bir köşenin x,y koordinatlarını yazdırıyoruz

        cv2.circle(resim, tuple(np.round(koseler[i]).astype("int")), 3, (255, 0, 0), 4)



cv2.imshow('Resim', resim)
cv2.waitKey(0)
cv2.destroyAllWindows()
