import pickle
import pandas as pd
from tensorflow.python.keras.layers import deserialize
from tensorflow.python.keras.saving import saving_utils

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

test_file = "data/test.csv"
raw_data_test = pd.read_csv(test_file)

text_x = raw_data_test.values.astype('float32')

text_x = text_x.reshape(text_x.shape[0], 28, 28,1)
text_x.shape

model = pickle.load(open('output/finalized_model.sav', 'rb'))
predictions = model.predict_classes(text_x, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("output/test-output.csv", index=False, header=True)
