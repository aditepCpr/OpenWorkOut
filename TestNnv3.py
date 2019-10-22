import pickle

try:

    f1 = open('knn.pkl', 'rb')
    stored_knn = pickle.load(f1)
    f1.close()
    print(stored_knn)
    # f2 = open('nn.pkl', 'rb')
    # stored_nn = pickle.load(f2)
    # f2.close()

    f3 = open('d_t.pkl', 'rb')
    stored_d_t = pickle.load(f3)
    f3.close()


    # print(stored_nn)
    print(stored_d_t)
except IOError as e:
    print(e)