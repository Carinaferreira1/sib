import numpy as np

class NeuralNetwork:

    def __init__(
        self,
        epochs=100,
        batch_size=16,
        optimizer=None,
        learning_rate=0.01,
        verbose=False,
        loss=None,
        metric=None,
        **kwargs
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.loss = loss
        self.metric = metric
        self.kwargs = kwargs

        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())

        if hasattr(layer, "weights"):
            optimizer = self.optimizer(
                learning_rate=self.learning_rate,
                **self.kwargs
            )
            layer.initialize(optimizer)

        self.layers.append(layer)

    def fit(self, dataset):
        X = dataset.X
        y = dataset.y

        n_samples = X.shape[0]

        self.history = {"loss": []}

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                output = X_batch

                for layer in self.layers:
                    output = layer.forward_propagation(output, training=True)

                batch_loss = self.loss.loss(y_batch, output)

                epoch_loss += batch_loss

                gradient = self.loss.derivative(y_batch, output)

                for layer in reversed(self.layers):
                    gradient = layer.backward_propagation(gradient)

            average_loss = epoch_loss / n_samples

            self.history["loss"].append(average_loss)

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{self.epochs} "
                    f"- Loss: {average_loss:.4f}"
                )

    def predict(self, dataset):
        output = dataset.X

        for layer in self.layers:
            output = layer.forward_propagation(output, training=False)

        return output

    def score(self, dataset):
        y_pred = self.predict(dataset)

        return self.metric(dataset.y, y_pred)