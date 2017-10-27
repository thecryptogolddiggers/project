file = open("ripple_price.csv", 'r')

out = open("ripple_price_new.csv", 'w')

file.readline()

for line in file:
    try:
        tokens = line.split("\"")
        # print(tokens)
        tokens[1] = tokens[1].replace(',', '')
        tokens[3] = tokens[3].replace(',', '')
        for t in tokens:
            out.write(t)
    except IndexError:
        pass

file.close()
out.close()