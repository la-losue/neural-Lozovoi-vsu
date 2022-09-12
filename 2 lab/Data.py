import json
import random


class DataHandler:
  def __init__(self, path: str = './input.json'):
    self.path = path

  def load_data(self):
    with open(self.path) as file:
      return json.load(file)

  def generate_training_dataframe(self, length: int = 100):
    data = self.load_data()
    res = []
    for _ in range(length):
      random_number = random.randint(0,9)
      res.append([ random_number, data[str(random_number)] ])
    return res
