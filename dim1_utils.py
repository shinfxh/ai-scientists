import numpy as np


vecs= ['x', 'y', 'Sx', 'Sy']
mats = ['Sxx', 'Syy', 'Sxy', 'Sxx^-1', 'Syy^-1', 'Sxy^-1']
targets = [22, 133] #('Syy^-1', 'Sx') and ('Syy^-1', 'Sxy', 'y')

def pick_terms(x):
    lv = len(vecs)
    lm = len(mats)
    if x <= lv - 1:
        return vecs[x]
    elif x <= lv + lm * lv - 1:
        x = x - lv
        return (mats[x//lv], vecs[x%lv])
    else:
        x = x - lv - lv * lm
        i1 = x//(lv*lm)
        x = x % (lv*lm)
        return (mats[i1], mats[x//lv], vecs[x%lv])

repeats = []
for i in range(172):
    s = pick_terms(i)
    check1 = 'Sxx' in s and 'Sxx^-1' in s
    check2 = 'Syy' in s and 'Syy^-1' in s
    check3 = 'Sxy' in s and 'Sxy^-1' in s
    if check1 or check2 or check3:
        repeats.append(i)