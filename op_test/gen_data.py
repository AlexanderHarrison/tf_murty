import numpy as np

m = 100
n = 80
c = (np.random.rand(1000, m, n) * 100.0).round() - 20
#mat = str(c).replace(".", ".0f32,")
#mat = str(mat).replace(",]", "]")
#mat = str(mat).replace("]\n [", "], [")
#mat = str(mat).replace(" ", "")
np.save("profiledata", c)
