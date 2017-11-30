# This is to pad the binary representations of words to make them be in the same length
# Here I'll pad 0

DIR = ''
fname = 'encode_keepmerging.txt'
fpath = DIR+fname
#read file
data = {}
f=open(fpath, 'rb')
lines = f.readlines()
rows = [eval(i.strip()) for i in lines]
for i in rows:
    data[i[0]] = i[1]
f.close()

max_length = 0
for code in data.values():
    if len(code) > max_length:
        max_length = len(code)

for key in data.keys():
    if len(data[key]) < max_length:
        for i in range(0,max_length-len(data[key])):
            data[key] = data[key] + '0'

f = open('binary_representation.txt', 'w')
for element in data.items():
    f.write(str(element))
    f.write('\n')
f.close()
