from Data import DataHandler
from Perceptron import Perceptron


if __name__ == '__main__':
  data_handler = DataHandler()
  perceptron = Perceptron(
    training_dataframe = data_handler.generate_training_dataframe(5)
  )

  perceptron.train()

  for i in range(10):
    print(f'Number {i} is:  ', perceptron.predict(data_handler.load_data()[str(i)]))
