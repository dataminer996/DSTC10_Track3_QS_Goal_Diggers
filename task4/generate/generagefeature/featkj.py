import numpy as np

a = [151.46861267089844, 209.1551055908203, 284.2559509277344, 302.611572265625]
image = [640,478]


def kj_feature(bbox,image):
  x1,y1,x2,y2 = list(bbox)
  imagex,imagey = list(image)
  newa = [x1/imagex,y1/imagey,x2/imagex,y2/imagey,(x2-x1)/imagex,(y2-y1)/imagey]
  return newa
print(kj_feature(a,image))
