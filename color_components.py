from PIL import Image

def get_color_components(path, img_name):
    im = Image.open(path+'/'+img_name+'.jpg')

    R, G, B = im.split()

    R.save(path+'/'+img_name+ "_R.jpg")
    G.save(path+'/'+img_name+ "_G.jpg")
    B.save(path+'/'+img_name+ "_B.jpg")

    H, S, V = im.convert("HSV").split()

    H.save(path+'/'+img_name+ "_H.jpg")
    S.save(path+'/'+img_name+ "_S.jpg")
    V.save(path+'/'+img_name+ "_V.jpg")

    C, M, Y, K = im.convert("CMYK").split()

    C.save(path+'/'+img_name+ "_C.jpg")
    M.save(path+'/'+img_name+ "_M.jpg")
    Y.save(path+'/'+img_name+ "_Y.jpg")

    Y, Cb, Cr = im.convert("YCbCr").split()

    Y.save(path+'/'+img_name+ "_Y.jpg")
    Cb.save(path+'/'+img_name+ "_Cb.jpg")
    Cr.save(path+'/'+img_name+ "_Cr.jpg")

    L, a, b = im.convert("LAB").split()

    L.save(path+'/'+img_name+ "_L.jpg")
    a.save(path+'/'+img_name+ "_a.jpg")
    b.save(path+'/'+img_name+ "_b.jpg")