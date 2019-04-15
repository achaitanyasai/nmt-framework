import sys
import tqdm
import string

fname = sys.argv[1]
f = open(fname)
g = open('/tmp/word-context.txt', 'w')

def check(s):
    if s in string.punctuation:
        return False
    return True

for line_num, _line in tqdm.tqdm(enumerate(f.readlines())):
    line = _line.strip().split()
    L = len(line)
    for i in range(L):
        cur = []
        for j in range(i - 5, i + 6):
            if j >= 0 and j < L and j != i and check(line[j]):
                cur.append(line[j])
        cur = ' '.join(cur)
        g.write('%d %d %s %s\n' % (line_num, i, line[i], cur))
f.close()
g.close()