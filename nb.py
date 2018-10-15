import numpy as np

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    vocabulary_size = matrix.shape[1]
    ###################
    state['phi_y'] = np.mean(category)

    # np.sum just take sum of entire matrix. That essentially means for each email, find a total of all the tokens and
    # add the total of each email when the spam label is 1
    state['phi_y1'] = (np.sum(matrix.T * category, axis=1) + 1)/(np.sum(matrix.T*category)+ N)
    #change the matrix values of 0 to 1 and 1 to 0
    category = -(category - 1)
    state['phi_y0'] = (np.sum(matrix.T * category, axis=1) + 1) / (np.sum(matrix.T*category)+ N)

    # mat1 = matrix[category == 1, :]
    # mat0 = matrix[category == 0, :]
    #
    # # documentation length, i.e. number of tokens in each document
    # mat1_doc_lens = mat1.sum(axis=1)
    # # yeq1 means "given y equals 1"
    # state['phi_yeq1'] = (mat1.sum(axis=0) + 1) / (np.sum(mat1_doc_lens) + vocabulary_size)
    #
    # mat0_doc_lens = mat0.sum(axis=1)
    # state['phi_yeq0'] = (mat0.sum(axis=0) + 1) / (np.sum(mat0_doc_lens) + vocabulary_size)
    #
    # state['phi'] = mat1.shape[0] / (mat1.shape[0] + mat0.shape[0])
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    log_phi_y1 = np.sum(np.log(state['phi_y1']) * matrix, axis=1)
    log_phi_y0 = np.sum(np.log(state['phi_y0']) * matrix, axis=1)
    phi_y = state['phi_y']

    ratio = np.exp(log_phi_y0 + np.log(1-phi_y) - log_phi_y1 - np.log(phi_y))
    probs = 1 / (1 + ratio)
    output[probs > 0.5] = 1
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)
    return error

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)
    evaluate(output, testCategory)

    #Part b
    p = np.log(state['phi_y1']/state['phi_y0'])
    indx = p.argsort()[-5:]
    indicative_tokens = [tokenlist[i] for i in indx]

    # print(indicative_tokens) result: ['valet', 'ebai', 'unsubscrib', 'spam', 'httpaddr']
    return

if __name__ == '__main__':
    main()
