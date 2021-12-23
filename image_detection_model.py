import requests
import scipy.misc
import numpy as np
import tensorflow as tf
import pandas as pd
import clases as cl

from six import BytesIO
from PIL import Image, ImageDraw, ImageFont #lib - Pillow
from six.moves.urllib.request import urlopen

def load_image_into_numpy_array(path):
  image = None
  if(path.startswith('http')):
    response = urlopen(path)
    image_data = response.read()
    image_data = BytesIO(image_data)
    image = Image.open(image_data)
  else:
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))

  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)

def detect(image, hub_model):
    image_np = load_image_into_numpy_array(image)
    ans = hub_model(image_np)
    detection_class_entities = map(int, ans["detection_classes"].numpy().tolist()[0])
    detection_scores = ans["detection_scores"].numpy().tolist()[0]
    res = dict(zip(detection_class_entities, detection_scores))

    df_1 = pd.DataFrame({'classes_id': list(res.keys()), 'scores': list(res.values())})
    df_2 = pd.DataFrame({'classes_id': list(cl.classes.keys()), 'name_class': list(cl.classes.values())})
    fin_df = df_1.merge(df_2, left_on='classes_id', right_on='classes_id')[['name_class', 'scores']]
    result = dict(zip(fin_df['name_class'], fin_df['scores']))
    return result
