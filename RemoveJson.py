import os

def removeJson():
    print('removeJson Start..')
    nameEx = 'cam'
    try:
        path = [os.path.join('dataSet/'+nameEx, f) for f in os.listdir('dataSet/'+nameEx)]
        for id in range(len(path)+1):
            file_path = ('dataSet/' + str(nameEx) + '/keypose.' + str(id) + ".json")
            # print('remove file = '+file_path)
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print("The file does not exist")
        print('removeJson Ok..')
    except IOError as e:
        print('removeJson :',e)


if __name__ == '__main__':
    print(removeJson())
