import numpy 

a = numpy.array([[0, 0,600 , 800], [0, 0, 50, 50]])
b = (600, 800)
numpy.save("imagebbox.npy",a)
numpy.save("imagesize.npy",b)

