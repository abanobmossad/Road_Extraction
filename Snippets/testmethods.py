l=['1234568.tif','1234.tif','214565.tif']
s='1234.tif'
f=s.split('.')[0]
if s in l:
    print(l.index(f+'.tif'))