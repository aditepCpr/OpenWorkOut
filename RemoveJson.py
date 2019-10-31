import os

def removeJson():
    print('removeJson Start..')
    nameEx = 'cam'
    try:
        path = [os.path.join('dataSet/'+nameEx, f) for f in os.listdir('dataSet/'+nameEx)]
        for id in path:
            if os.path.exists(id):
                os.remove(id)
            else:
                print("The file does not exist")
        print('removeJson Ok..')
    except IOError as e:
        print('removeJson :',e)

#
# if __name__ == '__main__':
#     print(removeJson())
