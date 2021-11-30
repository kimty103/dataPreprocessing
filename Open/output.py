import datetime

def setOutput(t, pred):
    now = datetime.datetime.now()
    f = open('Kong_first', 'w')
    f.write('%TEAM    {}\n'.format('first'))
    f.write('%DATE    {}\n'.format(now.strftime('%d-%H-%M-%S')))
    f.write('%TIME    {}\n'.format(t))
    f.write('%CASES   {}\n'.format(len(pred)))
    for i in range(1,len(pred)+1):
        f.write('T{}       {}\n'.format(i,pred[i-1]))
