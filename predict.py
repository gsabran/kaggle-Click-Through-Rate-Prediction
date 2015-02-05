import csv

def predict_on_test_data(classifier):
    with open('test.csv') as data, open('prediction.csv', 'wb') as pred:
        reader  = csv.reader(data)
        writer = csv.writer(pred)
        next(reader)
        writer.writerow(['id', 'click'])
        for row in reader:
            row_data = [float(i) for i in row[1:]]
            idx = row[0]
            y = classifier.predict_proba(np.array(row_data))
            writer.writerow([idx, y[0]])