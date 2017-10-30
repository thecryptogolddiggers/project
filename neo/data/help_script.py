import csv

# note: small bug causing most recent date row to not work

out = open("neo_price_new.csv", 'w')
csv_header = "date,open,high,low,close,volume,market_cap,increase_flag,percent_change\n"
out.write(csv_header)

with open('neo_price.csv', 'rU') as csvfile:
	neoreader = csv.reader(csvfile, delimiter=',')
	next(neoreader) # skip the first line    
	for i, row in enumerate(neoreader):
		row[5]=row[5].replace(',', '')
		row[6]=row[6].replace(',', '')

		x = (float(row[4])-float(row[1]))/(float(row[1])) * 100

		if x > 0:
			row.append(1)			
		else:
			row.append(0)	

		row.append(x)

		print row
		with open("neo_price_new.csv", "a") as fp:
		    wr = csv.writer(fp, dialect='excel')
		    wr.writerow(row)		

out.close()