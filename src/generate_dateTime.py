import datetime

start = datetime.datetime.strptime("2015-05-31", "%Y-%m-%d")
end = datetime.datetime.strptime("2019-05-16", "%Y-%m-%d")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]

for date in date_generated:
    print (date.strftime("%Y-%m-%d"))

