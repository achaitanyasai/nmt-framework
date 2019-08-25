from indictrans import Transliterator
import sys

fname = sys.argv[1]
src = sys.argv[2]
trg = sys.argv[3]
trn = Transliterator(source=src, target=trg, build_lookup=True)

f = open(fname)
x = f.read()
f.close()

def is_ascii_word(s):
    for i in s:
        if(('a' <= i <= 'z') or ('A' <= i <= 'Z')):
            return True
    return False

x = x.split('\n')
for i in x:
    cur = i.split(' ')
    a = []
    for word in cur:
        if(is_ascii_word(word)):
            res = trn.transform(word)
            a.append(res)
            # print(res.encode('utf-8'),
        else:
            a.append(word)
            # print word,
    print(' '.join(a))