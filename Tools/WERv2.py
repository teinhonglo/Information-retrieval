import sys
import numpy

def editDistance(r, h):
    '''
    This funciton is to calculate the edit distance of refernce sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0: 
                d[0][j] = j
            elif j == 0: 
                d[i][0] = i
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0: 
            break
        elif d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1] and x >= 1 and y >= 1: 
            list.append("e")
            x = x - 1
            y = y - 1
        elif d[x][y] == d[x][y-1]+1 and y >= 1:
            list.append("i")
            x = x
            y = y - 1
        elif d[x][y] == d[x-1][y-1]+1 and x >= 1 and y >= 1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]

def alignedPrint(list, r, h, result):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.
    
    Attributes:
        list   -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    print("REF:", end=" ")
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(" "*(len(h[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
                print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ")
            else:
                print(r[index1], end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(r[index], end=" ")
    print()
    print("HYP:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ")
            else:
                print(h[index2], end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(h[index], end=" ")
    print()
    print("EVA:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print("D" + " " * (len(r[index])-1), end=" ")
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print("I" + " " * (len(h[index])-1), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print("S" + " " * (len(r[index1])-1), end=" ")
            else:
                print("S" + " " * (len(h[index2])-1), end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print("C" + " " * (len(r[index])), end=" ")
    print()
    print("WER: " + result)

def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    # build the matrix
    d = editDistance(r, h)

    # find out the manipulation steps
    list = getStepList(r, h, d)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    result = str("%.2f" % result) + "%"
    alignedPrint(list, r, h, result)
    
    return float(d[len(r)][len(h)]), len(r)


if __name__ == '__main__':
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    word_errs, num_words = 0, 0

    with open(filename1,'rU') as f1, open(filename2, 'rU') as f2:
        for r, h in zip(f1,f2):
            r_list = r.split()
            h_list = h.split()
            r_uttid, r_list = r_list[0], r_list[1:]
            h_uttid, h_list = h_list[0], h_list[1:]
            print(r_uttid)
            word_err, num_word = wer(r_list, h_list)
            word_errs += word_err
            num_words += num_word
            print()
    
    result = word_errs / num_words * 100
    result = str("%.2f" % result) + "%"
    print("%WER (overall)", result)
            
    #r = file(filename1).read().split()
    #h = file(filename2).read().split()
    #wer(r, h)
