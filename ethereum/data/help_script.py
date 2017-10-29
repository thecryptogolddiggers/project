from datetime import datetime
import time

# datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')

def chomp(x):
    if x.endswith("\r\n"): return x[:-2]
    if x.endswith("\n"): return x[:-1]
    return x


# Get a list of the closing prices per day

file = open("ethereum_price.csv", 'r')
file.readline()

closingPrices = []
for line in file:
        tokens = line.split(",")
        close = tokens[4]
        closingPrices.append(close)
        print(close)
        # time.sleep(1)
file.close()

# Get a list of percentage changes

percentChanges = []
increaseFlags = []

percentChanges.append("JUNK")
increaseFlags.append("JUNK")

for x in range(0,len(closingPrices)-2):
    value = (float(closingPrices[x]) - float(closingPrices[x+1])) / (float(closingPrices[x+1])) * 100
    percentChanges.append( value)

    if value >= 0.0:
        increaseFlags.append(1)
    else:
        increaseFlags.append(0)





file = open("ethereum_price.csv", 'r')
file.readline()
out = open("ethereum_price_new.csv", 'w')

csv_header = "date,open,high,low,close,volume,market_cap,increase_flag,percent_change\n"
out.write(csv_header)

index = 0
for line in file:
    try:
        line = chomp(line)
        tokens = line.split(",")

        datetime_object = datetime.strptime(tokens[0], '%b %d %Y')

        days_since_epoch = (datetime_object - datetime(1970,1,1)).days


        tokens[0] = str(days_since_epoch)

        tokens[5] = tokens[5].replace("\"", "")
        tokens[5] = tokens[5].replace(",", "")


        tokens[6] = tokens[6].replace("\"", "")
        tokens[6] = tokens[6].replace(",", "")




        lineToPrint = ""
        for t in tokens:
            lineToPrint += (t)
            lineToPrint += (",")

        lineToPrint += str(increaseFlags[index])
        lineToPrint += ","
        lineToPrint += str(percentChanges[index])
        lineToPrint += "\n"

        index += 1

        out.write(lineToPrint)

    except IndexError:
        pass
        print("INDEX ERROR")
        exit()


    except IndexError:
        pass

file.close()
out.close()