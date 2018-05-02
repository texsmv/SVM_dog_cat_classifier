from sklearn import svm
import numpy as np
from preprocessing import *


clf = svm.SVC()
clf.fit(X, Y)
r_c = clf.predict([e for e in T_c])
r_c_t = len(r_c)
r_c_c = list(r_c).count(0)
r_d = clf.predict([e for e in T_d])
r_d_t = len(r_d)
r_d_c = list(r_d).count(1)
print(r_c_t)
print(r_c_c)
print(r_c)
print(r_d_t)
print(r_d_c)
print(r_d)
