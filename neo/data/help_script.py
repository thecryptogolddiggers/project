import csv
import pandas as pd
# note: small bug causing most recent date row to not work

out = open("neo_price_new.csv", 'w')
csv_header = "date,open,high,low,close,volume,market_cap,increase_flag,percent_change\n"
out.write(csv_header)
output=[]

with open('neo_price.csv', 'rU') as csvfile:
	neoreader = csv.reader(csvfile, delimiter=',')
	next(neoreader) # skip the first line    
	for i, row in enumerate(neoreader):
		row[5]=row[5].replace(',', '')
		row[6]=row[6].replace(',', '')

		if row[4] == '':
			row[4]=1
		if row[1] == '':
			row[1]=1

		x = (float(row[4])-float(row[1]))/(float(row[1])) * 100

		if x > 0:
			row.append(1)			
		else:
			row.append(0)	

		row.append(x)

		with open("neo_price_new.csv", "a") as fp:
		    wr = csv.writer(fp, dialect='excel')
		    wr.writerow(row)		

out.close()