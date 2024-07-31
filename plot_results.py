import matplotlib.pyplot as plt

class PlotResults:
    @staticmethod
    def plot(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(24, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Epoch')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Epoch')
        plt.show()

    @staticmethod
    def get_best_epoch(history):
        valid_acc = history.history['val_accuracy']
        best_epoch = valid_acc.index(max(valid_acc)) + 1
        best_acc = max(valid_acc)
        print('Best Validation Accuracy Score {:0.5f}, is for epoch {}'.format(best_acc, best_epoch))
        return best_epoch
