from Perceptron import Perceptron


class Network: 
  def __init__(self, training_dataframe, outputs: int):
    self.outputs = outputs
    self.perceptrons = [Perceptron(training_dataframe=training_dataframe) for _ in range(self.outputs)]

  def train(self):
    for i in range(self.outputs):
      self.perceptrons[i].train(i)

  def predict(self, input_vector: list[int]):
    answers = [perceptron.predict(input_vector) for perceptron in self.perceptrons]
    return answers.index(1)
