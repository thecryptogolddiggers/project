ripple = open('ripple_price.csv')
sp500 = open('sp500.csv')
out = open('sp500_new.csv', 'w')

ripple.readline()

header = sp500.readline()

out.write(header)

cur_bytes = len(header) + 1
prev_line = ""

for line in ripple:
    date = int(line.split(',')[0])
    sl = sp500.readline()
    print(sl)
    
    sdate = int(sl.split(',')[0])
    
    if date == sdate:
        out.write(sl)
        prev_line = sl
        cur_bytes += len(sl)+1

    else:
        out.write(prev_line)
        sp500.seek(cur_bytes, 0)

ripple.close()
sp500.close()
out.close()