import zipfile
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras

class Washer_Data(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths

    def __len__(self):
        #pass
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Return input correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)

        return x, y

def get_data_path():
    washer_ng_zip = 'sample_data/washer_ng.zip'
    washer_ok_zip = 'sample_data/washer_ok.zip'
    
    zip_ref_ok = zipfile.ZipFile(washer_ng_zip, 'r')
    zip_ref_ok.extractall('sample_data/')
    zip_ref_ok.close()
    
    zip_ref_ok = zipfile.ZipFile(washer_ok_zip, 'r')
    zip_ref_ok.extractall('sample_data/')
    zip_ref_ok.close()
    
    ok_img_file_names =  os.listdir('sample_data/washer_ok')
    ng_img_file_names = os.listdir('sample_data/washer_ng/kizu')
    #test_img_file_names = os.listdir('sample_data/washer_ng/sabi')
    
    train_path = []
    valid_path = []
    
    for (ok_img, ng_img) in zip(ok_img_file_names, ng_img_file_names[:30]):
        train_path.append('sample_data/washer_ok/'+ok_img)
        valid_path.append('sample_data/washer_ng/kizu/'+ng_img)
    
    return train_path, valid_path