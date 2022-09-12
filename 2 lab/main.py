from Data import DataHandler
from Network import Network


if __name__ == '__main__':
  data_handler = DataHandler()
  network = Network(
    training_dataframe = data_handler.generate_training_dataframe(),
    outputs = 10
  )

  network.train()

  while True:
    predict = input('Input number from 0 to 9:\n>>> ')
    input_vector = data_handler.load_data()[predict]
    print(f'Answer is: {network.predict(input_vector)}, input vector is: {input_vector}\n\n')
