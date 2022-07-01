class Activation:
    def __init__(self):
        self.cache = None
        self. dx = None

class Layer:
    def __init(self):
        self.cache = None
        self.W = None
        self.b = None
        self.Z = None
        self.dW = None
        self.db = None
        self.dx = None

    def update_weights(self, learning_rate):
        self.W = self.W - (learning_rate * self.dW)
        self.b = self.b - (learning_rate * self.db)
