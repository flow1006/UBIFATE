from faker import Faker
import random
import pandas as pd


class IntelligentCarInsuranceDataGenerator:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.fake = Faker()

    def generate_fake_data(self):
        data = []

        for _ in range(self.num_samples):
            age = random.randint(18, 65)
            income = random.randint(20000, 200000)
            credit_score = random.randint(300, 850)
            driving_experience = random.randint(1, 20)
            speeding_incidents = random.randint(0, 5)
            vehicle_type = random.choice(['Sedan', 'SUV', 'Truck'])
            coverage_type = random.choice(['Basic', 'Advanced', 'Premium'])
            claims_history = random.choices([0, 1], weights=[0.9, 0.1])[0]

            # Assume InsurancePremium is influenced by various factors
            insurance_premium = age * 0.01 + speeding_incidents * 0.02 + (claims_history * 0.1 if age < 25 else 0)

            row = [
                self.fake.unique.random_number(digits=6),
                age,
                income,
                credit_score,
                driving_experience,
                speeding_incidents,
                vehicle_type,
                coverage_type,
                claims_history,
                insurance_premium
            ]
            data.append(row)

        return data

    def generate_dataset(self, filename):
        data = self.generate_fake_data()
        df = pd.DataFrame(
            data,
            columns=[
                'ID',
                'Age',
                'Income',
                'CreditScore',
                'DrivingExperience',
                'SpeedingIncidents',
                'VehicleType',
                'CoverageType',
                'ClaimsHistory',
                'InsurancePremium'
            ]
        )
        df.to_csv(filename, index=False)
        print(f"已生成包含 {self.num_samples} 个样本的 {filename} 文件")


if __name__ == '__main__':
    # 使用 IntelligentCarInsuranceDataGenerator 类生成包含 10000 个样本的数据集
    generator = IntelligentCarInsuranceDataGenerator(10000)
    generator.generate_dataset('./car_data.csv')
