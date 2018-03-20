import cv2
import math



out=cv2.imread('Results/Second-result_road.png')
target=cv2.imread(r'E:\mass_roads\valid\map\23128930_15.tif')


tp=0
tn=0
fp=0
fn=0


for x in range(out.shape[0]):
    for y in range(out.shape[1]):

        if out[x,y][0]:
            if target[x+2,y+2][0]:
                tp+=1
            else:
                fp+=1
        else:
            if target[x+2,y+2][0]:
                fn+=1
            else:
                tn+=1


print('tp: ',tp,'\nfp: ',fp,'\nfn: ',fn,'\ntn: ',tn)



Recall=tp / (tp + fn)
Precision = tp / (tp + fp)


f1=2 * (Precision * Recall) / (Precision + Recall)
acc=((tp*tn)-(fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn++fp)*(tn+fn))

print('\nsum: ',(tp+fp+fn+tn),'\n\nall: ',1496*1496,'\n\nRecall: ',
      Recall,'\n\nPrecission: ',Precision,'\n\nfi: ',f1,'\n\nRcc: ',acc)




