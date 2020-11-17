import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

rawData = pd.read_csv("data/test.csv")
text_x = rawData.values.astype('float32')

text_x = text_x.reshape(text_x.shape[0], 28, 28,1)
text_x.shape

model = load_model('output/digit')
predictions = np.argmax(model.predict(text_x), axis=-1)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("output/test-output.csv", index=False, header=True)
