import csv

def read_csv(file, header=False):
    """return a parsed csv file in built-in dataframe"""
    data = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if header:
                continue
            line = [v_parse(val) for val in row]
            data.append(line)

    return data
