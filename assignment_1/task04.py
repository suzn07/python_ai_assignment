def my_split(sentence, sepChar):
    listOfSentence = []
    word = ""
    for s in sentence:
        if s == sepChar:
            listOfSentence.append(word)
            word = ""
        else:
            word += s
    listOfSentence.append(word)
    return listOfSentence

def my_join(items, sepChar):

    result = ""
    for i in range(len(items)):
        result += items[i]
        if i < len(items) - 1:
            result += sepChar
    return result

listOfSentence = input("Please enter sentence: ")

items = my_split(listOfSentence, " ")

print(my_join(items, ","))

for i in items:
    print(i)