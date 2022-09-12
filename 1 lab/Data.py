import json
import random

class DataHandler:
  def __init__(self, path: str = './input.json'):
    self.path = path

  def load_data(self):
    with open(self.path) as file:
      return json.load(file)

  # генерируем тренировочные данные
  # goal - число которое мы хотим угадывать
  def generate_training_dataframe(self, goal: int, length: int = 100):
    result = []
    data = self.load_data()
    for _ in range(length):
      random_number = random.randint(0,9)
      # определяем является ли рандомное число целевым
      goal_number = 0
      if (random_number == goal):
        goal_number = 1
      result.append([ goal_number, data[str(random_number)] ])
    return result
