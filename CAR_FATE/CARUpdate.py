import fate_data_loader
import fate_data_preprocessor
import fate_model_trainer
import fate_model_saver
import fate_model_loader
import fate_model_predictor

class IntelligentCarInsurancePricing:
    def __init__(self, guest_data_path, host_data_path, learning_rate=0.01, num_epochs=10, batch_size=32,
                 hidden_units=[64, 32], dropout_rate=0.2):
        self.guest_data_path = guest_data_path
        self.host_data_path = host_data_path
        self.guest_data = None
        self.host_data = None
        self.guest_train_data = None
        self.host_train_data = None
        self.guest_val_data = None
        self.host_val_data = None
        self.model = None
        self.model_param = {
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'hidden_units': hidden_units,
            'dropout_rate': dropout_rate
        }

    def load_data(self):
        self.guest_data = fate_data_loader.load(self.guest_data_path)
        self.host_data = fate_data_loader.load(self.host_data_path)

    def preprocess_data(self):
        self.guest_train_data = fate_data_preprocessor.preprocess(self.guest_data)
        self.host_train_data = fate_data_preprocessor.preprocess(self.host_data)

    def train_model(self):
        self.model = fate_model_trainer.train(self.guest_train_data, self.host_train_data, self.model_param)

    def save_model(self, model_path):
        fate_model_saver.save(self.model, model_path)

    def load_model(self, model_path):
        self.model = fate_model_loader.load(model_path)

    def predict(self, data):
        prediction_result = fate_model_predictor.predict(self.model, data)
        return prediction_result
