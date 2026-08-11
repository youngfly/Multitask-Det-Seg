import os

def mkdir(path):
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
        print("file makes successfully")
    else:
        print("file is exist")


pathq = '../dataset/seglabels/val/'
files = os.listdir(pathq)
for p in files:
    p_old = p
    p_new = p.split('_seg')[0] +'.png'
    print(p_old)
    print(p_new)
    # path = '/workspace/datapeo/datesets3/seg/'
    os.rename(pathq+p_old, pathq+p_new)
