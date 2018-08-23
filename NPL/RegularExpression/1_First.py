import re

sentence = "I was born in the year 1985"
sentence2 = ""
print (sentence)

# match any cahracter
print (re.match(r".*", sentence))
print (re.match(r".*", sentence2))

#same as * but ist says 1 or more
print (re.match(r".+", sentence))
print (re.match(r".+", sentence2))

print("Match of A_Za-z: ",re.match(r"[a-z]+",sentence))

print("Occurences of A_Za-z: ",re.search(r"[a-z]+",sentence))
print("Occurences of ab*: ",re.search(r"ab?",sentence))

re.sub(r"\d", "", sentence)
print (sentence)

