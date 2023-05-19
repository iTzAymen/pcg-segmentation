import pandas as pd
from os import walk

d = {'relative_path': [], 'classID': [], 'file_name': []}
class_weights = [0] * 4


def fillPaths(path, classID):
    for (dirpath, dirnames, filenames) in walk(path):
        relative_path = map(lambda self: dirpath + '/' + self, filenames)
        d['relative_path'].extend(relative_path)
        temp = [classID] * len(filenames)
        d['classID'].extend(temp)
        d['file_name'].extend(filenames)
        print(f"{path} [{len(filenames)}]")
        class_weights[classID] = 1/len(filenames)
        break


def getData(dirname='Dataset'):
    fillPaths(dirname + '/abnormal', 1)
    fillPaths(dirname + '/normal', 0)
    

    df = pd.DataFrame(data=d)
    df = df[df['file_name'].str.endswith('.wav')].reset_index(drop=True)

    return df

# df = getData(dirname='PASCAL')
# print(df)
# print(len(df))
# print(len(df[df['classID'] == 0]))
# print(len(df[df['classID'] == 1]))