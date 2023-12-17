import argparse
from federatedml import LinearRegression


class IntelligentCarInsurancePricingPredictor:
    def __init__(self, model_file):
        self.model_file = model_file
        self.model = None

    def set_model(self):
        # 加载模型
        self.model = LinearRegression()  # 使用适合回归任务的模型
        self.model.load_model(self.model_file)

    def predict_insurance_premium(self, input_data):
        # 调用模型进行保险费用预测
        predicted_premium = self.model.predict(input_data)
        # 在此可以根据需要进行后续处理
        return predicted_premium


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Intelligent Car Insurance Pricing Predictor')
    parser.add_argument('--data_file', type=str, help='Path to the input data file')
    parser.add_argument('--model_file', type=str, help='Path to the trained model file')
    args = parser.parse_args()

    # 获取命令行参数
    data_file = args.data_file
    model_file = args.model_file

    # 实例化 IntelligentCarInsurancePricingPredictor 类并加载模型
    predictor = IntelligentCarInsurancePricingPredictor(model_file)
    predictor.set_model()

    # 读取输入数据文件
    with open(data_file, 'r') as f:
        input_data = f.read().splitlines()

    # 调用模型进行预测
    predicted_premiums = []
    for data in input_data:
        # 根据实际情况解析输入数据，这里假设数据是以逗号分隔的
        input_data = [float(x) for x in data.split(',')]
        predicted_premium = predictor.predict_insurance_premium(input_data)
        predicted_premiums.append(predicted_premium)

    # 处理输出信息
    # 在此可以根据需要进行后续处理，例如打印输出、保存到文件等
    for premium in predicted_premiums:
        print("Predicted insurance premium:", premium)
