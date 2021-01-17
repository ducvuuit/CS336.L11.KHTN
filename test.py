def map(query, predict):
   mAp = 0.0
   num_correct = 0
   ap = 0.0
   for i, pre in enumerate(predict):
        print(i, pre)
        if query == pre:
            num_correct += 1
            ap += num_correct/(i+1)
   mAp = ap/num_correct
   return mAp
predict = [1, 0,0,1,1,0,0,0,0,0,0]
print(map(1, predict))