import sys
import tqdm
import string
import numpy as np

in_fname = sys.argv[1]
out_fname = sys.argv[2]

f = open(in_fname)
g = open(out_fname, 'w')

a = {}
mx_line_idx = 0

for _line in tqdm.tqdm(f.readlines()):
    line = _line.strip().split()
    line_idx = int(line[0])
    word_idx = int(line[1])
    probs = map(float, line[2:])
    vec_idx = np.argmax(probs)
    try:
        _ = a[line_idx]
    except KeyError:
        a[line_idx] = {}
    mx_line_idx = max(line_idx, mx_line_idx)
    a[line_idx][word_idx] = vec_idx

for i in tqdm.tqdm(range(mx_line_idx + 1)):
    s = ''
    for j in range(100):
        try:
            tmp = a[i][j]
            if s == '':
                s = s + '%d' % tmp
            else:
                s = s + ' %d' % tmp
        except KeyError:
            break
    g.write('%s\n' % s)
f.close()
g.close()