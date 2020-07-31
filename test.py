import pandas as pd
from tensorflow.keras.models import load_model

test_file = "data/test.csv"
raw_data_test = pd.read_csv(test_file)

text_x = raw_data_test.values.astype('float32')

text_x = text_x.reshape(text_x.shape[0], 28, 28,1)
text_x.shape

model = load_model('output/digit')
predictions = model.predict_classes(text_x, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("output/test-output.csv", index=False, header=True)
